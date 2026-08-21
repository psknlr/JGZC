"""Minimal HTTP adapters over the standard library.

Two shapes cover essentially every hosted model an operator is likely to point
this at: the OpenAI-compatible ``/chat/completions`` body (OpenAI, DeepSeek,
Qwen/DashScope compatible mode, vLLM, Ollama's OpenAI endpoint, most gateways)
and Anthropic's ``/v1/messages``. No SDK, no extra dependency — a hospital
deployment should not need a package index to start.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .base import LLMError

_TIMEOUT = 60


def _post(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:  # pragma: no cover
        raise LLMError(str(exc)) from exc


class OpenAICompatibleClient:
    """Any endpoint speaking the OpenAI chat-completions body."""

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        if not api_key:
            raise LLMError("缺少 API key")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.name = f"openai-compatible:{model}"

    def complete(self, system: str, prompt: str, *, max_tokens: int = 900) -> str:
        payload = _post(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}"},
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        try:
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"响应结构异常: {str(payload)[:200]}") from exc


class AnthropicClient:
    def __init__(self, model: str, api_key: str, base_url: str = "https://api.anthropic.com/v1") -> None:
        if not api_key:
            raise LLMError("缺少 API key")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.name = f"anthropic:{model}"

    def complete(self, system: str, prompt: str, *, max_tokens: int = 900) -> str:
        payload = _post(
            f"{self.base_url}/messages",
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        try:
            return "".join(
                block.get("text", "") for block in payload["content"] if block.get("type") == "text"
            )
        except (KeyError, TypeError) as exc:
            raise LLMError(f"响应结构异常: {str(payload)[:200]}") from exc


def build_client(provider: str | None = None, model: str | None = None) -> Any:
    """Build a client from arguments or environment, or the null client."""
    from .base import NullLLMClient

    provider = (provider or os.environ.get("OSTEOSARC_LLM_PROVIDER") or "").strip().lower()
    if not provider or provider == "none":
        return NullLLMClient()
    model = model or os.environ.get("OSTEOSARC_LLM_MODEL") or ""
    api_key = os.environ.get("OSTEOSARC_LLM_API_KEY", "")
    base_url = os.environ.get("OSTEOSARC_LLM_BASE_URL", "")

    if provider == "anthropic":
        return AnthropicClient(model or "claude-sonnet-4-5", api_key,
                               base_url or "https://api.anthropic.com/v1")
    if provider in ("openai", "openai_compatible", "compatible"):
        return OpenAICompatibleClient(model or "gpt-4o-mini", api_key,
                                      base_url or "https://api.openai.com/v1")
    raise LLMError(f"未知的 provider {provider!r}（可用: anthropic / openai）")


def describe(client: Any) -> str:
    return getattr(client, "name", type(client).__name__)
