"""provider: thin, OpenAI-compatible chat-completions client.

This is the only module (besides sources.fetch_job_log) that talks to a
network service carrying credentials. It is intentionally narrow:

- one POST to ``{base_url}/chat/completions``, non-streaming by default;
- reads base_url / model / timeout / max output tokens from the runtime
  config, and the API key from ``config.load_secrets`` (never logged);
- does NOT cache, retry speculatively beyond a hard count, or re-map the
  provider's response into any higher-level abstraction — callers get the
  raw message, tool_calls, finish_reason and usage back.

Credentials are only ever placed in the ``Authorization`` header. The value is
never printed, written to disk, or included in exceptions.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from config import DEFAULTS, RuntimeConfig, load_runtime_config, load_secrets


class ProviderError(Exception):
    """A compact failure carrying an error_type and a redacted message."""

    def __init__(self, error_type: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"error_type": self.error_type, "message": self.message, "details": self.details}


def _compose_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def build_chat_payload(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    model: str,
    max_tokens: int,
    temperature: float | None = None,
    stream: bool = False,
) -> dict:
    """The exact request body ``chat()`` sends — the single source of truth.

    The request-budget gate (agent._total_request_chars) measures THIS dict
    with serialize_payload, so the gate statistic and the wire payload cannot
    drift apart: one builder, one serializer, both call sites share them.
    """
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


def serialize_payload(payload: dict) -> str:
    """The exact wire serialization used for the POST body.

    json.dumps defaults (ensure_ascii=True) — deliberately the SAME call
    _post_json makes, so a character count over this string IS the payload
    size that goes on the wire.
    """
    return json.dumps(payload)


def _post_json(url: str, headers: dict, payload: dict, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=serialize_payload(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise ProviderError(
            "http_error",
            f"chat completions returned HTTP {exc.code}",
            {"http_status": exc.code, "body_prefix": detail},
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        raise ProviderError(
            "network_error",
            f"chat completions request failed: {reason or exc}",
        ) from exc
    except TimeoutError as exc:  # pragma: no cover - depends on runtime
        raise ProviderError("timeout", f"chat completions timed out after {timeout}s") from exc


class LLMProvider:
    def __init__(
        self,
        runtime: RuntimeConfig | None = None,
        secrets: dict[str, str] | None = None,
    ):
        self.runtime = runtime or load_runtime_config()
        self.secrets = secrets if secrets is not None else load_secrets()
        self.base_url = self.runtime.get("DEMO_A_LLM_BASE_URL") or DEFAULTS["DEMO_A_LLM_BASE_URL"]
        self.model = self.runtime.get("DEMO_A_LLM_MODEL") or DEFAULTS["DEMO_A_LLM_MODEL"]
        self.timeout = float(
            self.runtime.get("DEMO_A_LLM_TIMEOUT_SECONDS") or DEFAULTS["DEMO_A_LLM_TIMEOUT_SECONDS"]
        )
        self.max_output_tokens = self.runtime.get_int("DEMO_A_LLM_MAX_OUTPUT_TOKENS") or int(
            DEFAULTS["DEMO_A_LLM_MAX_OUTPUT_TOKENS"]
        )
        self.api_key = self.secrets.get("DEMO_A_LLM_API_KEY")
        if not self.api_key:
            raise ProviderError(
                "missing_credential",
                "DEMO_A_LLM_API_KEY is not set (check .env); cannot call the model",
                {"key": "DEMO_A_LLM_API_KEY"},
            )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "demo_a_foundation-provider",
        }

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
    ) -> dict:
        """One non-streaming chat completion.

        Returns a normalized dict:
          message       — the assistant message (content, tool_calls, and any
                          provider-specific continuation fields preserved)
          tool_calls    — normalized list of {id, name, arguments(dict)}
          finish_reason — string or None
          usage         — the provider's usage object as-is (None if absent)
          raw           — the full provider JSON (for caller inspection)

        A tool call whose ``arguments`` string does not parse as JSON is still
        surfaced with ``arguments=None`` and ``arguments_parse_error=True`` so
        the executor can report it as an explicit tool failure (not crash).
        """
        payload: dict = build_chat_payload(
            messages,
            tools,
            model=self.model,
            max_tokens=max_tokens or self.max_output_tokens,
            temperature=temperature,
            stream=stream,
        )

        status, data = _post_json(
            _compose_url(self.base_url), self._headers(), payload, self.timeout
        )

        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError):
            raise ProviderError(
                "bad_response",
                "chat completions response has no choices[0]",
                {"status": status},
            )

        raw_message = choice.get("message") or {}
        tool_calls = []
        for tc in raw_message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments")
            args = None
            parse_error = False
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except (ValueError, TypeError):
                    parse_error = True
            elif isinstance(args_raw, dict):
                args = args_raw
            tool_calls.append(
                {
                    "id": tc.get("id"),
                    "name": fn.get("name"),
                    "arguments": args,
                    "arguments_raw": args_raw,
                    "arguments_parse_error": parse_error,
                }
            )

        # Preserve provider-specific continuation fields on the message itself
        # (e.g. reasoning_content for reasoning models) so the caller can echo
        # them back verbatim when it continues a tool-call turn.
        message = dict(raw_message)
        message.setdefault("content", "")

        return {
            "message": message,
            "tool_calls": tool_calls,
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
            "raw": data,
        }


__all__ = [
    "LLMProvider",
    "ProviderError",
    "build_chat_payload",
    "serialize_payload",
    "load_runtime_config",
    "load_secrets",
]