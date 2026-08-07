from __future__ import annotations

from core.exceptions.base import AtlasError


class LLMError(AtlasError):
    """Raised when an LLM provider fails."""

    pass