"""Groq provider — free tier, ultra-fast inference, OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any

from tarno_backend.ai.provider import REQUEST_USER_AGENT, LLMProvider, LLMResponse, ToolCall

log = logging.getLogger(__name__)

_DEFAULT_URL = "https://api.groq.com/openai/v1"
_DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqProvider(LLMProvider):
    """Free cloud LLM via Groq — 320 tok/s, 1000 req/day, no credit card."""

    def __init__(
        self,
        api_key: str = "",
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_URL,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: float = 60.0,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        # Resolved by the caller (tarno.ai.factory) via SecretsVault, which
        # itself falls back to the GROQ_API_KEY env var - kept as a direct
        # fallback here too so GroqProvider() still works standalone.
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "GROQ_API_KEY nicht gesetzt. "
                "Erstelle einen kostenlosen Key auf https://console.groq.com/keys"
            )
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        self._max_retries = max_retries
        self._base_delay = base_delay
        log.info(
            "Groq Provider initialisiert (model=%s, timeout=%.1fs, retries=%d)",
            self._model,
            self._timeout,
            self._max_retries,
        )

    @property
    def name(self) -> str:
        return "groq"

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
        oai_messages = [{"role": "system", "content": system}]

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
            f"{self._base_url}/chat/completions",
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
            log.error("Groq API Fehler %d: %s", exc.code, body)
            if exc.code == 429:
                return LLMResponse(
                    text="Entschuldigung Sir, das Groq-Tageslimit ist erreicht. "
                         "Versuchen Sie es morgen erneut oder wechseln Sie zu Ollama.",
                    tool_calls=[],
                    is_error=True,
                )
            return LLMResponse(text=f"Groq API Fehler: {exc.code}", tool_calls=[], is_error=True)
        except (urllib.error.URLError, TimeoutError) as exc:
            log.error("Groq nicht erreichbar: %s", exc)
            return LLMResponse(text="Groq API nicht erreichbar.", tool_calls=[], is_error=True)

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
                        "tool_call_id": block["tool_use_id"],
                        "content": block.get("content", ""),
                    })
                elif block.get("type") == "tool_use":
                    return {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block["input"]),
                            },
                        }],
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
            ))

        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        return LLMResponse(text=text, tool_calls=tool_calls, raw=result)
