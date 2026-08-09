from __future__ import annotations

from infrastructure.llm.ollama_provider import (
    OllamaProvider,
)


class FakeOllamaClient:
    """
    Deterministic fake Ollama client.
    """

    def __init__(
        self,
        host: str,
        timeout: float,
    ) -> None:
        self.host = host
        self.timeout = timeout
        self.last_model: str | None = None
        self.last_prompt: str | None = None
        self.last_stream: bool | None = None

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        stream: bool,
    ) -> dict[str, str]:
        self.last_model = model
        self.last_prompt = prompt
        self.last_stream = stream

        return {
            "response": (
                "Atlas is a local knowledge platform."
            )
        }


class EmptyResponseClient(FakeOllamaClient):
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        stream: bool,
    ) -> dict[str, str]:
        return {
            "response": "   "
        }


class InvalidResponseClient(FakeOllamaClient):
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        stream: bool,
    ) -> dict[str, str]:
        return {
            "model": model
        }


class FailingOllamaClient(FakeOllamaClient):
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        stream: bool,
    ) -> dict[str, str]:
        raise RuntimeError(
            "connection refused"
        )


def test_provider_exposes_model_name() -> None:
    provider = OllamaProvider(
        model_name="qwen3:4b",
    )

    assert (
        provider.model_name
        == "qwen3:4b"
    )


def test_provider_exposes_base_url() -> None:
    provider = OllamaProvider(
        model_name="qwen3:4b",
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


def test_empty_base_url_is_rejected() -> None:
    try:
        OllamaProvider(
            model_name="qwen3:4b",
            base_url="   ",
        )
    except ValueError as error:
        assert "base_url" in str(error)
        return

    raise AssertionError(
        "Expected ValueError."
    )


def test_invalid_timeout_is_rejected() -> None:
    try:
        OllamaProvider(
            model_name="qwen3:4b",
            timeout=0,
        )
    except ValueError as error:
        assert "timeout" in str(error)
        return

    raise AssertionError(
        "Expected ValueError."
    )


def test_generate_uses_ollama_client(
    monkeypatch,
) -> None:
    created_clients: list[
        FakeOllamaClient
    ] = []

    def fake_client(
        *,
        host: str,
        timeout: float,
    ) -> FakeOllamaClient:
        client = FakeOllamaClient(
            host=host,
            timeout=timeout,
        )

        created_clients.append(
            client
        )

        return client

    monkeypatch.setattr(
        "infrastructure.llm.ollama_provider.ollama.Client",
        fake_client,
    )

    provider = OllamaProvider(
        model_name="qwen3:4b",
        base_url="http://127.0.0.1:11434",
        timeout=60,
    )

    result = provider.generate(
        "  What is Atlas?  "
    )

    assert (
        result
        == "Atlas is a local knowledge platform."
    )

    assert len(
        created_clients
    ) == 1

    client = created_clients[0]

    assert (
        client.host
        == "http://127.0.0.1:11434"
    )

    assert (
        client.timeout
        == 60
    )

    assert (
        client.last_model
        == "qwen3:4b"
    )

    assert (
        client.last_prompt
        == "What is Atlas?"
    )

    assert (
        client.last_stream
        is False
    )


def test_empty_prompt_is_rejected() -> None:
    provider = OllamaProvider(
        model_name="qwen3:4b",
    )

    try:
        provider.generate(
            "   "
        )
    except ValueError as error:
        assert "prompt" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError."
    )


def test_ollama_failure_becomes_runtime_error(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "infrastructure.llm.ollama_provider.ollama.Client",
        lambda **kwargs: FailingOllamaClient(
            host=kwargs["host"],
            timeout=kwargs["timeout"],
        ),
    )

    provider = OllamaProvider(
        model_name="qwen3:4b",
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


def test_invalid_response_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "infrastructure.llm.ollama_provider.ollama.Client",
        lambda **kwargs: InvalidResponseClient(
            host=kwargs["host"],
            timeout=kwargs["timeout"],
        ),
    )

    provider = OllamaProvider(
        model_name="qwen3:4b",
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


def test_empty_response_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "infrastructure.llm.ollama_provider.ollama.Client",
        lambda **kwargs: EmptyResponseClient(
            host=kwargs["host"],
            timeout=kwargs["timeout"],
        ),
    )

    provider = OllamaProvider(
        model_name="qwen3:4b",
    )

    try:
        provider.generate(
            "Hello"
        )
    except RuntimeError as error:
        assert "empty" in str(error).lower()
        return

    raise AssertionError(
        "Expected RuntimeError."
    )