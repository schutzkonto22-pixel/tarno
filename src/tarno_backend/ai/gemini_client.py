"""Google Gemini provider — free, 1500 req/day, OpenAI-compatible."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any

from tarno_backend.ai.provider import REQUEST_USER_AGENT, LLMProvider, LLMResponse, ToolCall, extract_reasoning

log = logging.getLogger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

# Aliases for old/invalid model IDs that may still be in saved configs or
# were previously listed in the model catalog (live fix: "gemini-3.5-flash"
# and "gemini-3.1-flash-lite" are not real Gemini API model IDs).
_GEMINI_MODEL_ALIASES: dict[str, str] = {
    "gemini-3.5-flash": "gemini-2.5-flash",
    "gemini-3.1-flash-lite": "gemini-2.5-flash-lite",
    "gemini-2.0-flash": "gemini-2.5-flash",
    "gemini-2.0-flash-lite": "gemini-2.5-flash-lite",
    "gemini-1.5-flash": "gemini-2.5-flash",
    "gemini-1.5-flash-8b": "gemini-2.5-flash-lite",
}

# Same class of bug as Mistral's _sanitize_mistral_tool_call_id (see
# mistral_client.py): tool_use ids in the shared, provider-agnostic
# conversation history may originate from a *different* provider and get
# replayed into a Gemini request after the user switches providers. Gemini's
# OpenAI-compatible endpoint has no documented strict id-format regex, so
# this is a defensive normalization (safe alphanumeric charset, bounded
# length) rather than a proven-necessary fix for a specific observed error -
# but keeps ids provider-neutral instead of passing arbitrary foreign
# formatting through unchecked. Deterministic (hash-based) so the same
# original id always maps to the same sanitized id on both the tool_use and
# tool_result sides of a pair.
_VALID_GEMINI_TOOL_CALL_ID = re.compile(r"^[a-zA-Z0-9_-]{1,40}$")


def _sanitize_gemini_tool_call_id(raw_id: str) -> str:
    if _VALID_GEMINI_TOOL_CALL_ID.match(raw_id or ""):
        return raw_id
    digest = hashlib.sha256((raw_id or "").encode("utf-8")).hexdigest()
    return f"call_{digest[:24]}"


class GeminiProvider(LLMProvider):
    """Free cloud LLM via Google Gemini API (OpenAI-compatible endpoint).

    1500 requests/day on free tier — no credit card required.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-2.0-flash",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: float = 60.0,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        # Resolved by the caller (tarno.ai.factory) via SecretsVault, which
        # itself falls back to the GEMINI_API_KEY env var - kept as a direct
        # fallback here too so GeminiProvider() still works standalone.
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY nicht gesetzt. "
                "Erstelle einen kostenlosen Key auf https://aistudio.google.com/apikey"
            )
        canonical = _GEMINI_MODEL_ALIASES.get(model, model)
        if canonical != model:
            log.warning("Gemini Modell-ID '%s' auf '%s' normalisiert", model, canonical)
        self._model = canonical
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        self._max_retries = max_retries
        self._base_delay = base_delay
        log.info(
            "Gemini Provider initialisiert (model=%s, timeout=%.1fs, retries=%d)",
            self._model,
            self._timeout,
            self._max_retries,
        )

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def supports_tools(self) -> bool:
        return True

    def send(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str | None = None,
        use_high_tier: bool = False,
    ) -> LLMResponse:
        oai_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

        for msg in messages:
            oai_messages.append(self._convert_message(msg))

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

        if tools:
            payload["tools"] = self._convert_tools(tools)
            payload["tool_choice"] = "auto"

        return self._call(payload)

    def send_tool_result(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str | None = None,
        use_high_tier: bool = False,
    ) -> LLMResponse:
        return self.send(messages, system, tools)

    def _call(self, payload: dict[str, Any]) -> LLMResponse:
        data = json.dumps(payload).encode("utf-8")
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
            ) as resp:
                result = json.loads(resp.read())
            return self._parse(result)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            log.error("Gemini API Fehler %d: %s", exc.code, body)
            if exc.code == 429:
                return LLMResponse(
                    text="Sir, das Gemini Rate-Limit ist vorübergehend erreicht. "
                         "Bitte warten Sie einen Moment.",
                    tool_calls=[],
                    is_error=True,
                )
            return LLMResponse(text=f"Gemini API Fehler: {exc.code}", tool_calls=[], is_error=True)
        except (urllib.error.URLError, TimeoutError) as exc:
            log.error("Gemini nicht erreichbar: %s", exc)
            return LLMResponse(text="Gemini API nicht erreichbar.", tool_calls=[], is_error=True)

    @staticmethod
    def _convert_message(msg: dict[str, Any]) -> dict[str, Any]:
        content = msg.get("content", "")
        role = msg.get("role", "user")

        if isinstance(content, str):
            return {"role": role, "content": content}

        if isinstance(content, list):
            parts: list[str] = []
            tool_results: list[dict[str, Any]] = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    parts.append(block["text"])
                elif block.get("type") == "tool_result":
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": _sanitize_gemini_tool_call_id(block["tool_use_id"]),
                        "content": block.get("content", ""),
                    })
                elif block.get("type") == "tool_use":
                    tool_call: dict[str, Any] = {
                        "id": _sanitize_gemini_tool_call_id(block["id"]),
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        },
                    }
                    # Gemini's "thinking" models require their own
                    # thought_signature to be echoed back verbatim on replay
                    # (see docs linked in the 400 error) - stored alongside
                    # the tool_use block when first captured in _parse().
                    # Without this, Gemini rejects its OWN historical
                    # function call the moment it appears again in context.
                    if block.get("provider_extra"):
                        tool_call["extra_content"] = block["provider_extra"]
                    return {
                        "role": "assistant",
                        # Preserve any text emitted alongside the tool call
                        # instead of silently discarding it (previously lost
                        # on every replay of a mixed text+tool_use turn).
                        "content": " ".join(parts) if parts else None,
                        "tool_calls": [tool_call],
                    }

            if tool_results:
                return tool_results[0]

            return {"role": role, "content": " ".join(parts) if parts else str(content)}

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
    def _parse(result: dict[str, Any]) -> LLMResponse:
        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = message.get("content", "") or ""
        tool_calls: list[ToolCall] = []

        for tc in (message.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(args)
            except json.JSONDecodeError:
                parsed_args = {}
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                input=parsed_args,
                raw_extra=tc.get("extra_content"),
            ))

        text, reasoning = extract_reasoning(text)
        return LLMResponse(text=text, tool_calls=tool_calls, raw=result, reasoning=reasoning)
