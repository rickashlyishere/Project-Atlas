from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    """
    Provider-agnostic interface for large language models.

    Atlas application code should depend on this interface
    rather than directly importing a specific LLM SDK.
    """

    @property
    def model_name(self) -> str:
        """
        Return the identifier of the model being used.
        """
        ...

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate a response from a prompt.
        """
        ...