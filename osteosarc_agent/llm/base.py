"""LLM transport contract.

The platform runs to completion with no model configured. A model, when
present, is a *narrator*: it may rephrase conclusions the deterministic pipeline
already reached. It never selects a recommendation, never resolves a conflict,
never clears a safety gate, and never produces a dose.
"""

from __future__ import annotations

from typing import Any, Protocol


class LLMError(RuntimeError):
    """Transport or configuration failure. Always degrades to the null path."""


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, prompt: str, *, max_tokens: int = 900) -> str: ...


class NullLLMClient:
    """The default. Returns nothing, so the platform stays deterministic."""

    name = "null"

    def complete(self, system: str, prompt: str, *, max_tokens: int = 900) -> str:
        return ""
