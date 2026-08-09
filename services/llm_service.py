from __future__ import annotations

from domain.llm.interfaces import LLMProvider


class LLMService:
    """
    Application-level service responsible for generating
    responses through an LLM provider.

    The service deliberately does not know whether the
    provider is local, remote, Ollama, or another backend.
    """

    def __init__(
        self,
        provider: LLMProvider,
    ) -> None:
        self.provider = provider

    @property
    def model_name(self) -> str:
        """
        Return the active provider model name.
        """

        return self.provider.model_name

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate an answer using the configured provider.
        """

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        response = self.provider.generate(
            prompt
        )

        response = response.strip()

        if not response:
            raise ValueError(
                "LLM provider returned an empty response."
            )

        return response