from __future__ import annotations

from services.context_assembler import (
    ContextAssembler,
)
from services.llm_service import LLMService
from services.prompt_builder import (
    GroundedPromptBuilder,
)
from services.rag_service import (
    RAGService,
)
from services.search_service import (
    SearchService,
)


class FakeSearchService(SearchService):
    """
    Deterministic search service for RAG tests.
    """

    def __init__(
        self,
        results: list[dict],
    ) -> None:
        self.results = results
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        self.last_query = query
        self.last_top_k = top_k

        return self.results[:top_k]


class FakeLLMProvider:
    """
    Deterministic LLM provider for RAG tests.
    """

    def __init__(
        self,
        response: str = (
            "The document states that Python "
            "is a programming language. [Source 1]"
        ),
    ) -> None:
        self.response = response
        self.last_prompt: str | None = None

    @property
    def model_name(self) -> str:
        return "fake-rag-model"

    def generate(
        self,
        prompt: str,
    ) -> str:
        self.last_prompt = prompt
        return self.response


def create_rag_service(
    search_results: list[dict],
) -> tuple[
    RAGService,
    FakeSearchService,
    FakeLLMProvider,
]:
    search_service = FakeSearchService(
        search_results
    )

    llm_provider = FakeLLMProvider()

    llm_service = LLMService(
        llm_provider
    )

    rag_service = RAGService(
        search_service=search_service,
        context_assembler=ContextAssembler(),
        prompt_builder=GroundedPromptBuilder(),
        llm_service=llm_service,
    )

    return (
        rag_service,
        search_service,
        llm_provider,
    )


def make_result(
    text: str = "Python is a programming language.",
    score: float = 0.95,
) -> dict:
    return {
        "chunk_id": "chunk-1",
        "text": text,
        "score": score,
        "page_number": 1,
        "document_id": "document-1",
        "filename": "python.txt",
        "chunk_type": "fixed",
    }


def test_rag_service_returns_answer() -> None:
    service, _, _ = create_rag_service(
        [
            make_result()
        ]
    )

    response = service.answer(
        "What is Python?"
    )

    assert (
        response.answer
        == (
            "The document states that Python "
            "is a programming language. [Source 1]"
        )
    )


def test_rag_service_returns_sources() -> None:
    service, _, _ = create_rag_service(
        [
            make_result()
        ]
    )

    response = service.answer(
        "What is Python?"
    )

    assert len(
        response.sources
    ) == 1

    assert (
        response.sources[0].filename
        == "python.txt"
    )

    assert (
        response.sources[0].page_number
        == 1
    )


def test_rag_service_searches_question() -> None:
    service, search_service, _ = create_rag_service(
        [
            make_result()
        ]
    )

    service.answer(
        "  What is Python?  ",
        top_k=3,
    )

    assert (
        search_service.last_query
        == "What is Python?"
    )

    assert (
        search_service.last_top_k
        == 3
    )


def test_rag_service_sends_grounded_prompt_to_llm() -> None:
    service, _, llm_provider = create_rag_service(
        [
            make_result()
        ]
    )

    service.answer(
        "What is Python?"
    )

    assert (
        llm_provider.last_prompt
        is not None
    )

    assert (
        "What is Python?"
        in llm_provider.last_prompt
    )

    assert (
        "Python is a programming language."
        in llm_provider.last_prompt
    )

    assert (
        "Do not invent facts"
        in llm_provider.last_prompt
    )


def test_rag_service_exposes_model_name() -> None:
    service, _, _ = create_rag_service(
        [
            make_result()
        ]
    )

    assert (
        service.model_name
        == "fake-rag-model"
    )


def test_empty_question_is_rejected() -> None:
    service, _, _ = create_rag_service(
        [
            make_result()
        ]
    )

    try:
        service.answer("")
    except ValueError as error:
        assert "question" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError."
    )


def test_invalid_top_k_is_rejected() -> None:
    service, _, _ = create_rag_service(
        [
            make_result()
        ]
    )

    try:
        service.answer(
            "What is Python?",
            top_k=0,
        )
    except ValueError as error:
        assert "top_k" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError."
    )


def test_no_context_is_rejected() -> None:
    service, _, _ = create_rag_service(
        []
    )

    try:
        service.answer(
            "What is Python?"
        )
    except ValueError as error:
        assert "context" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError."
    )


def test_rag_service_preserves_search_results() -> None:
    results = [
        make_result(
            text="First result.",
            score=0.95,
        ),
        make_result(
            text="Second result.",
            score=0.85,
        ),
    ]

    service, _, _ = create_rag_service(
        results
    )

    response = service.answer(
        "Tell me about Python.",
        top_k=2,
    )

    assert (
        response.search_results
        == results
    )


def test_rag_service_returns_assembled_context() -> None:
    service, _, _ = create_rag_service(
        [
            make_result()
        ]
    )

    response = service.answer(
        "What is Python?"
    )

    assert (
        "Python is a programming language."
        in response.context.text
    )

    assert len(
        response.context.sources
    ) == 1