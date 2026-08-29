"""Mistral AI provider — free, 1B tokens/month, 1 req/s, OpenAI-compatible."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import string
import urllib.error
import urllib.request
from typing import Any, Iterator

from tarno_backend.ai.provider import (
    REQUEST_USER_AGENT,
    LLMProvider,
    LLMResponse,
    ProviderCapabilities,
    RateLimiter,
    StreamingDelta,
    ToolCall,
    extract_reasoning,
)

log = logging.getLogger(__name__)

_BASE_URL = "https://api.mistral.ai/v1"

# Mistral's own API occasionally returns a tool_call.id that violates its
# OWN validation regex on the next request ("Tool call id was X but must be
# a-z, A-Z, 0-9, with a length of 9" - a real, live-observed Mistral-side
# quirk). The same regex also bites when a tool_use id minted by a
# *different* provider (e.g. an 8-char Groq-style id) sits in the shared,
# provider-agnostic conversation history and gets replayed into a Mistral
# request after the user switches providers - confirmed live via tarno.log
# ("Tool call id was yiolww12 but must be ... length of 9"). Applied at BOTH
# the receive path (_parse, guarding Mistral's own malformed responses) and
# the send path (_convert_message, guarding foreign-origin ids replayed from
# history) - deterministic (hash-based, not random) so the same original id
# always sanitizes to the same output on both the tool_use and tool_result
# sides of a pair, keeping them matched no matter which site converts first.
_VALID_MISTRAL_TOOL_CALL_ID = re.compile(r"^[a-zA-Z0-9]{9}$")


def _sanitize_mistral_tool_call_id(raw_id: str) -> str:
    if _VALID_MISTRAL_TOOL_CALL_ID.match(raw_id or ""):
        return raw_id
    digest = hashlib.sha256((raw_id or "").encode("utf-8")).hexdigest()
    alphabet = string.ascii_letters + string.digits
    return "".join(alphabet[int(digest[i * 2:i * 2 + 2], 16) % len(alphabet)] for i in range(9))

# Context-window size per model, for the UI's token-usage indicator.
# Sizes per Mistral's published model docs (2026-07-13). Falls back to
# 32000 (mistral-small's window) for unlisted/future model IDs so the
# indicator degrades gracefully instead of showing nothing.
_CONTEXT_WINDOWS: dict[str, int] = {
    "mistral-small-latest": 32_000,
    "mistral-small-2506": 32_000,
    "mistral-large-latest": 131_072,
    "ministral-3b-2512": 32_000,
    "labs-leanstral-1-5": 256_000,
}
_DEFAULT_CONTEXT_WINDOW = 32_000

# Models that can produce visible reasoning in some form (research stand
# 2026-07-20) - reasoning_effort is silently dropped/ignored for any other
# model instead of erroring, so an outdated list degrades gracefully rather
# than breaking requests.
_REASONING_CAPABLE_MODELS: frozenset[str] = frozenset({
    "magistral-small-latest", "magistral-medium-latest", "mistral-medium-latest",
})

# Found live: magistral models reason unconditionally by design and reject
# the reasoning_effort field outright ("Mistral Streaming Fehler 400:
# reasoning_effort is not enabled for this model") - the dial only applies
# to non-magistral reasoning-capable models (e.g. mistral-medium-latest).
_MAGISTRAL_MODEL_PREFIX = "magistral-"

# Used to auto-redirect when "Denken" is on but the selected chat model
# (e.g. mistral-large-latest) can't do reasoning_effort at all - keeps the
# toggle meaningful regardless of which model is otherwise selected.
_DEFAULT_REASONING_MODEL = "magistral-medium-latest"

# Published per-model request throughput limits for Mistral's API
# (requests per second, 2026-07-24). Missing or future IDs fall back to a
# conservative default so the limiter degrades gracefully instead of failing.
_MISTRAL_RPS_BY_MODEL: dict[str, float] = {
    "codestral-2508": 2.08,
    "codestral-latest": 2.08,
    "codestral-embed": 1.00,
    "devstral-2512": 0.83,
    "devstral-latest": 0.83,
    "labs-leanstral-1-5-1": 0.63,
    "labs-leanstral-1-5": 0.63,
    "magistral-medium-2509": 0.08,
    "magistral-medium-latest": 0.08,
    "magistral-small-2509": 0.03,
    "magistral-small-latest": 0.03,
    "ministral-14b-2512": 0.50,
    "ministral-3b-2512": 12.50,
    "ministral-8b-2512": 3.13,
    "mistral-embed-2312": 1.00,
    "mistral-large-2512": 0.07,
    "mistral-large-latest": 0.07,
    "mistral-medium-2505": 0.42,
    "mistral-medium-2508": 0.38,
    "mistral-medium-latest": 0.83,
    "mistral-moderation-2603": 1.67,
    "mistral-small-2506": 5.00,
    "mistral-small-2603": 0.83,
    "mistral-small-latest": 0.83,
    "open-mistral-nemo": 0.50,
    "voxtral-mini-2507": 1.00,
    "voxtral-mini-2602": 1.00,
    "voxtral-mini-transcribe-realtime-2602": 1.00,
    "voxtral-mini-tts-2603": 1.00,
}
_MISTRAL_RPS_DEFAULT = 0.50


class MistralProvider(LLMProvider):
    """Free cloud LLM via Mistral AI API.

    1 billion tokens/month, 1 request/second on free tier.
    OpenAI-compatible API — no credit card required.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "mistral-small-latest",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        frequency_penalty: float = 0.4,
        presence_penalty: float = 0.4,
        timeout: float = 60.0,
        max_retries: int = 3,
        base_delay: float = 1.0,
        difficulty_tiers: dict[str, str] | None = None,
        difficulty_enabled: bool = False,
        rate_limit_rps: float = 0.0,
    ) -> None:
        # Resolved by the caller (tarno.ai.factory) via SecretsVault, which
        # itself falls back to the MISTRAL_API_KEY env var - kept as a direct
        # fallback here too so MistralProvider() still works standalone
        # (e.g. in tests) without going through the factory.
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY ist nicht gesetzt. TARNO braucht einen Mistral-API-Key, "
                "damit er antworten kann.\n\n"
                "So bekommst du einen Key:\n"
                "1. Rufe https://console.mistral.ai/ auf und melde dich an (oder registriere dich).\n"
                "2. Klicke im Menü auf \"API Keys\" und dann auf \"Create API Key\".\n"
                "3. Kopiere den angezeigten Key.\n"
                "4. Lege ihn als Windows-Umgebungsvariable MISTRAL_API_KEY an.\n"
                "5. Starte TARNO neu.\n\n"
                "Der Key wird ausschließlich aus der Umgebungsvariable gelesen; "
                "er wird nicht in der TARNO-Konfiguration gespeichert."
            )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty
        self._timeout = timeout
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._difficulty_enabled = difficulty_enabled
        self._difficulty_tiers = difficulty_tiers or {}
        # 0.0 means "auto" -> lookup per-model RPS. Any positive value forces
        # that fixed RPS for all Mistral calls.
        self._rate_limit_rps = rate_limit_rps
        self._rate_limiters: dict[str, RateLimiter] = {}
        log.info(
            "Mistral Provider initialisiert (model=%s, timeout=%.1fs, retries=%d, difficulty_tiers=%s)",
            self._model,
            self._timeout,
            self._max_retries,
            "an" if self._difficulty_enabled else "aus",
        )

    @property
    def name(self) -> str:
        return "mistral"

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(streaming=True, tool_calls=True, reasoning_effort=True)

    def _apply_rate_limit(self, model: str) -> None:
        """Set the active rate limiter for the chosen model before a request.

        A fixed override (config rate_limit_rps > 0) always wins. Otherwise we
        look up the published per-model RPS and fall back to a safe default.
        """
        if self._rate_limit_rps > 0:
            rps = self._rate_limit_rps
        else:
            rps = _MISTRAL_RPS_BY_MODEL.get(model, _MISTRAL_RPS_DEFAULT)
        if model not in self._rate_limiters:
            self._rate_limiters[model] = RateLimiter(rps)
        self._rate_limiter = self._rate_limiters[model]

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str | None = None,
        use_high_tier: bool = False,
        stream: bool = False,
    ) -> dict[str, Any]:
        oai_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

        for msg in messages:
            oai_messages.append(self._convert_message(msg))

        model = self._model
        if use_high_tier and self._difficulty_enabled and self._difficulty_tiers.get("high"):
            model = self._difficulty_tiers["high"]
            log.debug("Starkes Modell manuell angefordert -> '%s'", model)

        # Found live: "Denken" (reasoning) toggled on with a non-reasoning
        # model selected (e.g. mistral-large-latest) silently dropped
        # reasoning_effort below with no visible reasoning ever produced -
        # the UI has no way to know a given model can't do this, so redirect
        # to a real reasoning-capable model instead of silently no-op'ing.
        if reasoning_effort and model not in _REASONING_CAPABLE_MODELS:
            log.info(
                "'%s' unterstützt kein reasoning_effort - weiche für diese Anfrage auf '%s' aus",
                model, _DEFAULT_REASONING_MODEL,
            )
            model = _DEFAULT_REASONING_MODEL

        log.info("Mistral-Anfrage: model=%s", model)

        payload: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "frequency_penalty": self._frequency_penalty,
            "presence_penalty": self._presence_penalty,
            "stream": stream,
        }

        if tools:
            payload["tools"] = self._convert_tools(tools)
            payload["tool_choice"] = "auto"

        if reasoning_effort and not model.startswith(_MAGISTRAL_MODEL_PREFIX):
            # model is guaranteed reasoning-capable here - either it already
            # was, or the redirect above just switched it to one. Magistral
            # models are excluded: they reason unconditionally and the API
            # rejects reasoning_effort for them (see note above).
            payload["reasoning_effort"] = reasoning_effort

        return payload

    def send(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str | None = None,
        use_high_tier: bool = False,
    ) -> LLMResponse:
        payload = self._build_payload(messages, system, tools, reasoning_effort, use_high_tier, stream=False)
        return self._call(payload)

    def send_stream(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str | None = None,
        use_high_tier: bool = False,
    ) -> Iterator[StreamingDelta]:
        payload = self._build_payload(messages, system, tools, reasoning_effort, use_high_tier, stream=True)
        yield from self._call_stream(payload)

    @staticmethod
    def _last_user_text(messages: list[dict[str, Any]]) -> str:
        """Extract the plain text of the most recent user turn, for difficulty
        classification. Handles both plain-string content and the
        Anthropic-style content-block list format used elsewhere in the app."""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
                if parts:
                    return " ".join(parts)
        return ""

    @staticmethod
    def _classify_difficulty(text: str) -> str:
        """Cheap, local heuristic - no extra LLM call. Pure word-count only:
        a prior keyword-based "high" auto-trigger (matching tool-adjacent
        words like "Datei" anywhere in the text) used to mis-fire on ordinary
        messages ("...aufwendige Datei über Vögel" routed to a slow
        specialized model for a plain chat request). "high" is no longer
        reachable from this classifier at all - only via the explicit
        manual "Starkes Modell"-toggle (see send()'s use_high_tier)."""
        word_count = len(text.split())
        if word_count > 12:
            return "medium"
        return "low"

    def send_tool_result(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str | None = None,
        use_high_tier: bool = False,
    ) -> LLMResponse:
        return self.send(messages, system, tools, reasoning_effort=reasoning_effort, use_high_tier=use_high_tier)

    def _call(self, payload: dict[str, Any], allow_fallback: bool = True) -> LLMResponse:
        data = json.dumps(payload).encode("utf-8")
        self._apply_rate_limit(payload.get("model", self._model))
        req = urllib.request.Request(
            f"{_BASE_URL}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": REQUEST_USER_AGENT,
            },
        )

        try:
            with self._http_request_with_retry(
                req,
                timeout=self._timeout,
                max_retries=self._max_retries,
                base_delay=self._base_delay,
                max_429_attempts=0,
            ) as resp:
                result = json.loads(resp.read())
            return self._parse(result, str(payload.get("model", self._model)))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            log.error("Mistral API Fehler %d: %s", exc.code, body)
            if exc.code == 429 and allow_fallback:
                # Rate limit exhausted on this model: if difficulty tiers
                # are configured, one shot at the most generous-quota
                # tier ("low", highest RPS/TPM) before giving up entirely.
                low_model = self._difficulty_tiers.get("low")
                if low_model and payload.get("model") != low_model:
                    log.warning(
                        "Rate-Limit auf '%s' erschöpft, weiche auf Low-Tier-Modell '%s' aus",
                        payload.get("model"), low_model,
                    )
                    fallback_payload = dict(payload, model=low_model)
                    return self._call(fallback_payload, allow_fallback=False)
                return LLMResponse(
                    text="Sir, das Mistral Rate-Limit ist weiterhin aktiv. "
                         "Bitte warten Sie einen Moment.",
                    tool_calls=[],
                    is_error=True,
                )
            if exc.code == 403 and allow_fallback:
                # Labs-Modelle (z. B. das "high"-Difficulty-Tier) muessen
                # pro Mistral-Organisation manuell freigeschaltet werden;
                # ohne Freischaltung liefert die API hier hart 403 -
                # unabhaengig von Nachrichtenlaenge/-inhalt. Statt die
                # ganze Anfrage scheitern zu lassen, weicht das auf das
                # naechstniedrigere, garantiert freigeschaltete Tier aus.
                error_type = ""
                try:
                    error_type = json.loads(body).get("type", "")
                except (json.JSONDecodeError, AttributeError):
                    pass
                if error_type == "labs_not_enabled":
                    fallback_model = (
                        self._difficulty_tiers.get("medium")
                        or self._difficulty_tiers.get("low")
                    )
                    if fallback_model and payload.get("model") != fallback_model:
                        log.warning(
                            "Labs-Modell '%s' nicht freigeschaltet (403), weiche auf '%s' aus",
                            payload.get("model"), fallback_model,
                        )
                        fallback_payload = dict(payload, model=fallback_model)
                        return self._call(fallback_payload, allow_fallback=False)
            return LLMResponse(text=f"Mistral API Fehler: {exc.code}", tool_calls=[], is_error=True)
        except (urllib.error.URLError, TimeoutError) as exc:
            log.error("Mistral nicht erreichbar: %s", exc)
            return LLMResponse(text="Mistral API nicht erreichbar.", tool_calls=[], is_error=True)

    def _call_stream(self, payload: dict[str, Any]) -> Iterator[StreamingDelta]:
        data = json.dumps(payload).encode("utf-8")
        self._apply_rate_limit(payload.get("model", self._model))
        req = urllib.request.Request(
            f"{_BASE_URL}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": REQUEST_USER_AGENT,
                "Accept": "text/event-stream",
            },
        )

        try:
            with self._http_request_with_retry(
                req,
                timeout=self._timeout,
                max_retries=self._max_retries,
                base_delay=self._base_delay,
                max_429_attempts=0,
            ) as resp:
                tool_acc: dict[int, dict[str, Any]] = {}
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        yield StreamingDelta(is_finished=True)
                        return
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}

                    content = delta.get("content") or ""
                    if content:
                        yield StreamingDelta(text=content)

                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        acc = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            acc["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            acc["name"] = fn["name"]
                        if fn.get("arguments"):
                            acc["arguments"] += fn["arguments"]

                    finish = choice.get("finish_reason")
                    if finish in ("tool_calls", "stop"):
                        if finish == "tool_calls" and tool_acc:
                            for idx in sorted(tool_acc.keys()):
                                acc = tool_acc[idx]
                                args: dict[str, Any] = {}
                                if acc["arguments"]:
                                    try:
                                        args = json.loads(acc["arguments"])
                                    except json.JSONDecodeError:
                                        args = {}
                                yield StreamingDelta(
                                    tool_call=ToolCall(
                                        id=_sanitize_mistral_tool_call_id(acc["id"]),
                                        name=acc["name"],
                                        input=args,
                                    )
                                )
                        yield StreamingDelta(is_finished=True)
                        return
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            log.error("Mistral Streaming Fehler %d: %s", exc.code, body)
            yield StreamingDelta(text=f"Mistral API Fehler: {exc.code}", is_finished=True, is_error=True)
        except (urllib.error.URLError, TimeoutError) as exc:
            log.error("Mistral Streaming nicht erreichbar: %s", exc)
            yield StreamingDelta(text="Mistral API nicht erreichbar.", is_finished=True, is_error=True)

    @staticmethod
    def _convert_message(msg: dict[str, Any]) -> dict[str, Any]:
        content = msg.get("content", "")
        role = msg.get("role", "user")

        if isinstance(content, str):
            # Mistral rejects assistant messages with empty content and no
            # tool_calls ("must have either content or tool_calls, but not
            # none") - guard here so a stale/empty stored turn can never
            # brick the whole conversation on replay.
            if role == "assistant" and not content:
                content = "..."
            return {"role": role, "content": content}

        if isinstance(content, list):
            text_parts: list[str] = []
            image_blocks: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "image_url":
                    # Ohne diesen Zweig wuerden Bild-Bloecke stillschweigend
                    # verworfen (fielen durch alle folgenden elif-Zweige
                    # durch) - ein hochgeladenes Bild kaeme dann NIE bei
                    # Mistral an, obwohl die UI es erfolgreich gesendet hat.
                    image_blocks.append(block)
                elif block.get("type") == "tool_result":
                    tool_content = block.get("content", "")
                    if not isinstance(tool_content, str):
                        tool_content = json.dumps(tool_content, ensure_ascii=False)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": _sanitize_mistral_tool_call_id(block["tool_use_id"]),
                        "content": tool_content,
                    })
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": _sanitize_mistral_tool_call_id(block["id"]),
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        },
                    })

            if tool_calls:
                return {
                    "role": "assistant",
                    "content": " ".join(text_parts) if text_parts else "",
                    "tool_calls": tool_calls,
                }

            if tool_results:
                return tool_results[0]

            if image_blocks:
                # Multi-Part-Content statt kollabiertem String, damit das
                # Bild tatsaechlich mitgeschickt wird (Mistrals Chat-
                # Completions-API akzeptiert dasselbe OpenAI-style
                # {"type":"image_url","image_url":{"url":...}}-Blockformat).
                parts: list[dict[str, Any]] = []
                if text_parts:
                    parts.append({"type": "text", "text": " ".join(text_parts)})
                parts.extend(image_blocks)
                return {"role": role, "content": parts}

            joined = " ".join(text_parts) if text_parts else str(content)
            if role == "assistant" and not joined.strip():
                # Same guard as above: an assistant text-only message must
                # not resolve to empty content with no tool_calls.
                joined = "..."
            return {"role": role, "content": joined}

        return {"role": role, "content": str(content)}

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

    @staticmethod
    def _split_content_chunks(content: Any) -> tuple[str, str | None]:
        """reasoning_effort="high" makes Mistral return message.content as a
        chunk list (ThinkChunk/TextChunk) instead of a plain string - each
        thinking chunk holds its own nested text sub-blocks. Separates
        visible text from reasoning instead of the plain-string
        extract_reasoning() path, which only handles inline <think> tags."""
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "thinking":
                for sub in block.get("thinking", []) or []:
                    if isinstance(sub, dict) and sub.get("text"):
                        reasoning_parts.append(sub["text"])
            elif block_type == "text" and block.get("text"):
                text_parts.append(block["text"])
        return " ".join(text_parts).strip(), (" ".join(reasoning_parts).strip() or None)

    @staticmethod
    def _parse(result: dict[str, Any], model: str = "") -> LLMResponse:
        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        raw_content = message.get("content", "") or ""
        if isinstance(raw_content, list):
            text, reasoning = MistralProvider._split_content_chunks(raw_content)
        else:
            text, reasoning = extract_reasoning(raw_content)
        tool_calls: list[ToolCall] = []

        for tc in (message.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(args)
            except json.JSONDecodeError:
                parsed_args = {}
            tool_calls.append(ToolCall(
                id=_sanitize_mistral_tool_call_id(tc.get("id", "")),
                name=fn.get("name", ""),
                input=parsed_args,
            ))

        usage = result.get("usage") or {}
        tokens_used = usage.get("total_tokens")
        context_window = _CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT_WINDOW) if tokens_used is not None else None

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            raw=result,
            tokens_used=tokens_used,
            context_window=context_window,
            reasoning=reasoning,
        )
