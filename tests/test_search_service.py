from domain.embeddings import EmbeddingProvider

from services.search_service import SearchService
from services.vector_search_service import (
    VectorSearchService,
)


class FakeEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic embedding provider for search tests.
    """

    @property
    def dimension(self) -> int:
        return 3

    def embed_text(
        self,
        text: str,
    ) -> list[float]:

        text = text.lower()

        if "python" in text:
            return [
                1.0,
                0.0,
                0.0,
            ]

        if "linux" in text:
            return [
                0.0,
                1.0,
                0.0,
            ]

        return [
            0.0,
            0.0,
            1.0,
        ]

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        return [
            self.embed_text(text)
            for text in texts
        ]


class FakeEmbeddingService:
    """
    Minimal embedding service used for search tests.
    """

    def __init__(self) -> None:
        self.provider = FakeEmbeddingProvider()

    @property
    def dimension(self) -> int:
        return self.provider.dimension


class FakeEmbeddingRepository:
    """
    In-memory embedding repository for search tests.
    """

    def __init__(self) -> None:
        self.records: list[dict] = [
            {
                "embedding_id": "embedding-python",
                "chunk_id": "chunk-python",
                "model_name": "test-model",
                "dimension": 3,
                "vector": [
                    1.0,
                    0.0,
                    0.0,
                ],
                "created_at": "2026-01-01T00:00:00",
                "text": (
                    "Python is a programming language."
                ),
                "page_number": 1,
                "chunk_type": "fixed",
                "document_id": "document-1",
                "filename": "python.txt",
            },
            {
                "embedding_id": "embedding-linux",
                "chunk_id": "chunk-linux",
                "model_name": "test-model",
                "dimension": 3,
                "vector": [
                    0.0,
                    1.0,
                    0.0,
                ],
                "created_at": "2026-01-01T00:00:01",
                "text": (
                    "Linux is an operating system."
                ),
                "page_number": 2,
                "chunk_type": "fixed",
                "document_id": "document-1",
                "filename": "python.txt",
            },
            {
                "embedding_id": "embedding-other",
                "chunk_id": "chunk-other",
                "model_name": "test-model",
                "dimension": 3,
                "vector": [
                    0.0,
                    0.0,
                    1.0,
                ],
                "created_at": "2026-01-01T00:00:02",
                "text": (
                    "Triangles have three sides."
                ),
                "page_number": 1,
                "chunk_type": "fixed",
                "document_id": "document-2",
                "filename": "math.txt",
            },
        ]

    def get_all_for_search(
        self,
    ) -> list[dict]:
        return self.records

    def get_by_document(
        self,
        document_id: str,
    ) -> list[dict]:
        return [
            record
            for record in self.records
            if record["document_id"] == document_id
        ]


def create_search_service() -> SearchService:
    return SearchService(
        embedding_service=FakeEmbeddingService(),
        embedding_repository=FakeEmbeddingRepository(),
        vector_search_service=VectorSearchService(),
    )


def test_search_returns_most_relevant_result() -> None:
    service = create_search_service()

    results = service.search(
        "Tell me about Python programming.",
        top_k=3,
    )

    assert len(results) == 3

    assert (
        results[0]["chunk_id"]
        == "chunk-python"
    )

    assert results[0]["score"] == 1.0


def test_search_returns_document_metadata() -> None:
    service = create_search_service()

    results = service.search(
        "Python",
        top_k=1,
    )

    result = results[0]

    assert result["filename"] == "python.txt"
    assert result["page_number"] == 1
    assert result["document_id"] == "document-1"


def test_search_respects_top_k() -> None:
    service = create_search_service()

    results = service.search(
        "Python",
        top_k=1,
    )

    assert len(results) == 1


def test_search_empty_query_is_rejected() -> None:
    service = create_search_service()

    try:
        service.search("")
    except ValueError as error:
        assert "empty" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError for empty query."
    )


def test_search_document_limits_candidates() -> None:
    service = create_search_service()

    results = service.search_document(
        document_id="document-1",
        query="Linux",
        top_k=5,
    )

    assert len(results) == 2

    assert all(
        result["document_id"] == "document-1"
        for result in results
    )

    assert (
        results[0]["chunk_id"]
        == "chunk-linux"
    )


def test_search_empty_document_returns_no_results() -> None:
    service = create_search_service()

    results = service.search_document(
        document_id="does-not-exist",
        query="Python",
    )

    assert results == []