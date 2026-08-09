from __future__ import annotations

from typing import Any

import ollama

from domain.llm.interfaces import LLMProvider


class OllamaProvider:
    """
    LLM provider backed by a local Ollama server.

    Uses the official Ollama Python client.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:
        model_name = model_name.strip()
        base_url = base_url.strip().rstrip("/")

        if not model_name:
            raise ValueError(
                "model_name cannot be empty."
            )

        if not base_url:
            raise ValueError(
                "base_url cannot be empty."
            )

        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero."
            )

        self._model_name = model_name
        self._base_url = base_url
        self._timeout = timeout

        self._client = ollama.Client(
            host=self._base_url,
            timeout=self._timeout,
        )

    @property
    def model_name(self) -> str:
        """
        Return the configured Ollama model name.
        """

        return self._model_name

    @property
    def base_url(self) -> str:
        """
        Return the configured Ollama server URL.
        """

        return self._base_url

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate a response using Ollama.
        """

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        try:
            response = self._client.generate(
                model=self._model_name,
                prompt=prompt,
                stream=False,
            )

        except Exception as error:
            raise RuntimeError(
                "Could not generate a response from "
                f"Ollama model '{self._model_name}'. "
                "Make sure Ollama is running and the "
                "model is available."
            ) from error

        response_text: Any = response.get(
            "response"
        )

        if not isinstance(
            response_text,
            str,
        ):
            raise RuntimeError(
                "Ollama response did not contain "
                "a valid 'response' field."
            )

        response_text = response_text.strip()

        if not response_text:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return response_text