from __future__ import annotations

from services.llm_service import LLMService


class FakeLLMProvider:
    """
    Deterministic LLM provider for unit tests.
    """

    def __init__(
        self,
        response: str = "This is a test answer.",
    ) -> None:
        self.response = response
        self.last_prompt: str | None = None

    @property
    def model_name(self) -> str:
        return "fake-test-model"

    def generate(
        self,
        prompt: str,
    ) -> str:
        self.last_prompt = prompt
        return self.response


def test_llm_service_returns_provider_response() -> None:
    provider = FakeLLMProvider(
        response="Atlas test response."
    )

    service = LLMService(
        provider
    )

    result = service.generate(
        "What is Atlas?"
    )

    assert (
        result
        == "Atlas test response."
    )


def test_llm_service_exposes_model_name() -> None:
    provider = FakeLLMProvider()

    service = LLMService(
        provider
    )

    assert (
        service.model_name
        == "fake-test-model"
    )


def test_llm_service_passes_prompt_to_provider() -> None:
    provider = FakeLLMProvider()

    service = LLMService(
        provider
    )

    service.generate(
        "  Explain this document.  "
    )

    assert (
        provider.last_prompt
        == "Explain this document."
    )


def test_empty_prompt_is_rejected() -> None:
    provider = FakeLLMProvider()

    service = LLMService(
        provider
    )

    try:
        service.generate("")
    except ValueError as error:
        assert "prompt" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError for empty prompt."
    )


def test_whitespace_prompt_is_rejected() -> None:
    provider = FakeLLMProvider()

    service = LLMService(
        provider
    )

    try:
        service.generate("   ")
    except ValueError as error:
        assert "prompt" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError for whitespace prompt."
    )


def test_empty_provider_response_is_rejected() -> None:
    provider = FakeLLMProvider(
        response="   "
    )

    service = LLMService(
        provider
    )

    try:
        service.generate(
            "Give me an answer."
        )
    except ValueError as error:
        assert "empty" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError for empty provider response."
    )