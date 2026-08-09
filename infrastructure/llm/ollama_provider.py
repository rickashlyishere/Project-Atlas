from __future__ import annotations

import json
import urllib.error
import urllib.request

from domain.llm.interfaces import LLMProvider


class OllamaProvider:
    """
    LLM provider backed by a local Ollama server.

    Atlas communicates with Ollama through its HTTP API.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:

        model_name = model_name.strip()
        base_url = base_url.rstrip("/")

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

    @property
    def model_name(self) -> str:
        """
        Return the configured Ollama model name.
        """

        return self._model_name

    @property
    def base_url(self) -> str:
        """
        Return the configured Ollama base URL.
        """

        return self._base_url

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate a response from Ollama.
        """

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
        }

        request = urllib.request.Request(
            url=f"{self._base_url}/api/generate",
            data=json.dumps(
                payload
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=self._timeout,
            ) as response:

                raw_response = response.read()

        except urllib.error.URLError as error:

            raise RuntimeError(
                "Could not connect to Ollama at "
                f"{self._base_url}. "
                "Make sure Ollama is running."
            ) from error

        except TimeoutError as error:

            raise RuntimeError(
                "The Ollama request timed out."
            ) from error

        try:

            response_data = json.loads(
                raw_response.decode("utf-8")
            )

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:

            raise RuntimeError(
                "Ollama returned an invalid JSON response."
            ) from error

        response_text = response_data.get(
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