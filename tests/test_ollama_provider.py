from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from infrastructure.llm.ollama_provider import OllamaProvider


DEFAULT_MODEL = "qwen3:4b"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 300.0


def create_provider(
    model_name: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> OllamaProvider:
    return OllamaProvider(
        model_name=model_name,
        base_url=base_url,
        timeout=timeout,
    )


def test_provider_exposes_model_name() -> None:
    provider = create_provider(
        model_name="llama3.2",
    )

    assert provider.model_name == "llama3.2"


def test_provider_exposes_base_url() -> None:
    provider = create_provider(
        base_url="http://localhost:11434",
    )

    assert (
        provider.base_url
        == "http://localhost:11434"
    )


def test_empty_model_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="model_name"):
        create_provider(
            model_name="   ",
        )


def test_empty_base_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="base_url"):
        create_provider(
            base_url="   ",
        )


def test_invalid_timeout_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="timeout",
    ):
        create_provider(
            timeout=0,
        )


def test_generate_uses_ollama_client() -> None:
    provider = create_provider()

    mock_client = MagicMock()

    mock_client.generate.return_value = {
        "response": "Hello from Ollama."
    }

    provider._client = mock_client

    result = provider.generate(
        "Say hello."
    )

    assert result == "Hello from Ollama."

    mock_client.generate.assert_called_once_with(
        model=DEFAULT_MODEL,
        prompt="Say hello.",
        stream=False,
    )


def test_empty_prompt_is_rejected() -> None:
    provider = create_provider()

    with pytest.raises(
        ValueError,
        match="Prompt",
    ):
        provider.generate("   ")


def test_ollama_failure_becomes_runtime_error() -> None:
    provider = create_provider()

    mock_client = MagicMock()

    mock_client.generate.side_effect = (
        RuntimeError("connection failed")
    )

    provider._client = mock_client

    with pytest.raises(
        RuntimeError,
        match="Could not generate",
    ):
        provider.generate(
            "Hello."
        )


def test_invalid_response_is_rejected() -> None:
    provider = create_provider()

    mock_client = MagicMock()

    mock_client.generate.return_value = {
        "response": None
    }

    provider._client = mock_client

    with pytest.raises(
        RuntimeError,
        match="valid 'response'",
    ):
        provider.generate(
            "Hello."
        )


def test_empty_response_is_rejected() -> None:
    provider = create_provider()

    mock_client = MagicMock()

    mock_client.generate.return_value = {
        "response": "   "
    }

    provider._client = mock_client

    with pytest.raises(
        RuntimeError,
        match="empty response",
    ):
        provider.generate(
            "Hello."
        )


def test_generate_strips_prompt() -> None:
    provider = create_provider()

    mock_client = MagicMock()

    mock_client.generate.return_value = {
        "response": "Hello."
    }

    provider._client = mock_client

    provider.generate(
        "   Hello there.   "
    )

    mock_client.generate.assert_called_once_with(
        model=DEFAULT_MODEL,
        prompt="Hello there.",
        stream=False,
    )


def test_generate_strips_response() -> None:
    provider = create_provider()

    mock_client = MagicMock()

    mock_client.generate.return_value = {
        "response": "   Hello.   "
    }

    provider._client = mock_client

    result = provider.generate(
        "Hello."
    )

    assert result == "Hello."


@patch(
    "infrastructure.llm.ollama_provider.ollama.Client"
)
def test_provider_configures_ollama_client(
    mock_client_class: MagicMock,
) -> None:
    create_provider(
        model_name="llama3.2",
        base_url="http://127.0.0.1:11434",
        timeout=300.0,
    )

    mock_client_class.assert_called_once_with(
        host="http://127.0.0.1:11434",
        timeout=300.0,
    )


def test_provider_accepts_custom_timeout() -> None:
    provider = create_provider(
        timeout=600.0,
    )

    assert provider._timeout == 600.0


def test_provider_normalizes_base_url() -> None:
    provider = create_provider(
        base_url="http://127.0.0.1:11434///",
    )

    assert (
        provider.base_url
        == "http://127.0.0.1:11434"
    )


def test_provider_rejects_negative_timeout() -> None:
    with pytest.raises(
        ValueError,
        match="timeout",
    ):
        create_provider(
            timeout=-1,
        )
