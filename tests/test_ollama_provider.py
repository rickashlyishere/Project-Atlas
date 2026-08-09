from __future__ import annotations

import json
import urllib.error

from infrastructure.llm.ollama_provider import (
    OllamaProvider,
)


class FakeResponse:
    """
    Minimal fake HTTP response.
    """

    def __init__(
        self,
        payload: dict,
    ) -> None:

        self.payload = payload

    def __enter__(
        self,
    ) -> "FakeResponse":

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        return None

    def read(
        self,
    ) -> bytes:

        return json.dumps(
            self.payload
        ).encode("utf-8")


def test_provider_exposes_model_name() -> None:

    provider = OllamaProvider(
        model_name="test-model",
    )

    assert (
        provider.model_name
        == "test-model"
    )


def test_provider_exposes_base_url() -> None:

    provider = OllamaProvider(
        model_name="test-model",
        base_url="http://localhost:11434/",
    )

    assert (
        provider.base_url
        == "http://localhost:11434"
    )


def test_empty_model_name_is_rejected() -> None:

    try:

        OllamaProvider(
            model_name="   ",
        )

    except ValueError as error:

        assert "model_name" in str(error)
        return

    raise AssertionError(
        "Expected ValueError."
    )


def test_empty_prompt_is_rejected() -> None:

    provider = OllamaProvider(
        model_name="test-model",
    )

    try:

        provider.generate("   ")

    except ValueError as error:

        assert "prompt" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError."
    )


def test_generate_returns_ollama_response(
    monkeypatch,
) -> None:

    def fake_urlopen(
        request,
        timeout,
    ) -> FakeResponse:

        assert (
            request.full_url
            == "http://127.0.0.1:11434/api/generate"
        )

        payload = json.loads(
            request.data.decode("utf-8")
        )

        assert (
            payload["model"]
            == "test-model"
        )

        assert (
            payload["prompt"]
            == "What is Atlas?"
        )

        assert (
            payload["stream"]
            is False
        )

        return FakeResponse(
            {
                "response": "Atlas is a knowledge platform."
            }
        )

    monkeypatch.setattr(
        "urllib.request.urlopen",
        fake_urlopen,
    )

    provider = OllamaProvider(
        model_name="test-model",
    )

    result = provider.generate(
        "  What is Atlas?  "
    )

    assert (
        result
        == "Atlas is a knowledge platform."
    )


def test_connection_error_becomes_runtime_error(
    monkeypatch,
) -> None:

    def fake_urlopen(
        request,
        timeout,
    ):

        raise urllib.error.URLError(
            "connection refused"
        )

    monkeypatch.setattr(
        "urllib.request.urlopen",
        fake_urlopen,
    )

    provider = OllamaProvider(
        model_name="test-model",
    )

    try:

        provider.generate(
            "Hello"
        )

    except RuntimeError as error:

        assert "Ollama" in str(error)
        return

    raise AssertionError(
        "Expected RuntimeError."
    )


def test_invalid_json_becomes_runtime_error(
    monkeypatch,
) -> None:

    class InvalidResponse(FakeResponse):

        def read(self) -> bytes:

            return b"not valid json"

    def fake_urlopen(
        request,
        timeout,
    ) -> InvalidResponse:

        return InvalidResponse({})

    monkeypatch.setattr(
        "urllib.request.urlopen",
        fake_urlopen,
    )

    provider = OllamaProvider(
        model_name="test-model",
    )

    try:

        provider.generate(
            "Hello"
        )

    except RuntimeError as error:

        assert "JSON" in str(error)
        return

    raise AssertionError(
        "Expected RuntimeError."
    )


def test_missing_response_field_is_rejected(
    monkeypatch,
) -> None:

    def fake_urlopen(
        request,
        timeout,
    ) -> FakeResponse:

        return FakeResponse(
            {
                "model": "test-model",
            }
        )

    monkeypatch.setattr(
        "urllib.request.urlopen",
        fake_urlopen,
    )

    provider = OllamaProvider(
        model_name="test-model",
    )

    try:

        provider.generate(
            "Hello"
        )

    except RuntimeError as error:

        assert "response" in str(error).lower()
        return

    raise AssertionError(
        "Expected RuntimeError."
    )