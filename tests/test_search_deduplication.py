from __future__ import annotations

from typing import Any

from services.search_service import SearchService


class FakeEmbeddingProvider:
    @property
    def dimension(self) -> int:
        return 3

    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.provider = FakeEmbeddingProvider()

    @property
    def dimension(self) -> int:
        return self.provider.dimension


class FakeEmbeddingRepository:
    def __init__(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        self.records = records

    def get_all_for_search(
        self,
    ) -> list[dict[str, Any]]:
        return self.records

    def get_by_document(
        self,
        document_id: str,
    ) -> list[dict[str, Any]]:
        return [
            record
            for record in self.records
            if record["document_id"] == document_id
        ]


class FakeVectorSearchService:
    def __init__(self) -> None:
        self.last_candidates: list[
            dict[str, Any]
        ] = []

        self.last_top_k: int | None = None

    def search(
        self,
        query_vector: list[float],
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        self.last_candidates = candidates
        self.last_top_k = top_k

        return candidates[:top_k]


def create_record(
    *,
    chunk_id: str,
    document_id: str,
    filename: str,
    page_number: int,
    text: str,
) -> dict[str, Any]:
    return {
        "embedding_id": f"embedding-{chunk_id}",
        "chunk_id": chunk_id,
        "model_name": "test-model",
        "dimension": 3,
        "vector": [
            1.0,
            0.0,
            0.0,
        ],
        "created_at": "2026-01-01T00:00:00",
        "text": text,
        "page_number": page_number,
        "chunk_type": "fixed",
        "document_id": document_id,
        "filename": filename,
    }


def create_service(
    records: list[dict[str, Any]],
) -> tuple[
    SearchService,
    FakeVectorSearchService,
]:
    vector_service = (
        FakeVectorSearchService()
    )

    service = SearchService(
        embedding_service=FakeEmbeddingService(),
        embedding_repository=FakeEmbeddingRepository(
            records
        ),
        vector_search_service=vector_service,
    )

    return service, vector_service


def test_search_removes_duplicate_document_chunks() -> None:
    records = [
        create_record(
            chunk_id="chunk-1",
            document_id="document-1",
            filename="Rithvik.pdf",
            page_number=1,
            text="Rithvik is interested in Python.",
        ),
        create_record(
            chunk_id="chunk-2",
            document_id="document-2",
            filename="Rithvik.pdf",
            page_number=1,
            text="Rithvik is interested in Python.",
        ),
        create_record(
            chunk_id="chunk-3",
            document_id="document-3",
            filename="Rithvik.pdf",
            page_number=1,
            text="Rithvik is interested in Python.",
        ),
    ]

    service, _ = create_service(records)

    results = service.search(
        "Rithvik interests",
        top_k=5,
    )

    assert len(results) == 1

    assert (
        results[0]["chunk_id"]
        == "chunk-1"
    )


def test_search_preserves_distinct_chunks() -> None:
    records = [
        create_record(
            chunk_id="chunk-1",
            document_id="document-1",
            filename="Rithvik.pdf",
            page_number=1,
            text="Rithvik likes Python.",
        ),
        create_record(
            chunk_id="chunk-2",
            document_id="document-1",
            filename="Rithvik.pdf",
            page_number=2,
            text="Rithvik works with Linux.",
        ),
    ]

    service, _ = create_service(records)

    results = service.search(
        "Rithvik interests",
        top_k=5,
    )

    assert len(results) == 2


def test_search_ignores_whitespace_differences() -> None:
    records = [
        create_record(
            chunk_id="chunk-1",
            document_id="document-1",
            filename="Rithvik.pdf",
            page_number=1,
            text="Rithvik likes Python.",
        ),
        create_record(
            chunk_id="chunk-2",
            document_id="document-2",
            filename="Rithvik.pdf",
            page_number=1,
            text="  Rithvik   likes   Python.  ",
        ),
    ]

    service, _ = create_service(records)

    results = service.search(
        "Python",
        top_k=5,
    )

    assert len(results) == 1


def test_search_keeps_same_text_from_different_documents() -> None:
    records = [
        create_record(
            chunk_id="chunk-1",
            document_id="document-1",
            filename="one.pdf",
            page_number=1,
            text="Python is useful.",
        ),
        create_record(
            chunk_id="chunk-2",
            document_id="document-2",
            filename="two.pdf",
            page_number=1,
            text="Python is useful.",
        ),
    ]

    service, _ = create_service(records)

    results = service.search(
        "Python",
        top_k=5,
    )

    assert len(results) == 2


def test_search_top_k_applies_after_deduplication() -> None:
    records = [
        create_record(
            chunk_id="duplicate-1",
            document_id="document-1",
            filename="same.pdf",
            page_number=1,
            text="Duplicate content.",
        ),
        create_record(
            chunk_id="duplicate-2",
            document_id="document-2",
            filename="same.pdf",
            page_number=1,
            text="Duplicate content.",
        ),
        create_record(
            chunk_id="unique-1",
            document_id="document-3",
            filename="other.pdf",
            page_number=1,
            text="Unique content.",
        ),
    ]

    service, vector_service = create_service(
        records
    )

    results = service.search(
        "content",
        top_k=2,
    )

    assert len(results) == 2

    assert vector_service.last_top_k == 2

    assert (
        vector_service.last_candidates
    )

    assert len(
        vector_service.last_candidates
    ) == 2


def test_search_document_removes_duplicates() -> None:
    records = [
        create_record(
            chunk_id="chunk-1",
            document_id="document-1",
            filename="Rithvik.pdf",
            page_number=1,
            text="Rithvik likes Python.",
        ),
        create_record(
            chunk_id="chunk-2",
            document_id="document-1",
            filename="Rithvik.pdf",
            page_number=1,
            text="Rithvik likes Python.",
        ),
        create_record(
            chunk_id="chunk-3",
            document_id="document-1",
            filename="Rithvik.pdf",
            page_number=2,
            text="Rithvik uses Linux.",
        ),
    ]

    service, _ = create_service(records)

    results = service.search_document(
        document_id="document-1",
        query="Rithvik",
        top_k=5,
    )

    assert len(results) == 2


def test_search_rejects_empty_query() -> None:
    service, _ = create_service([])

    try:
        service.search("")

    except ValueError as error:
        assert "empty" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError."
    )


def test_search_rejects_invalid_top_k() -> None:
    service, _ = create_service([])

    try:
        service.search(
            "Python",
            top_k=0,
        )

    except ValueError as error:
        assert "top_k" in str(error)
        return

    raise AssertionError(
        "Expected ValueError."
    )


def test_search_document_rejects_empty_document_id() -> None:
    service, _ = create_service([])

    try:
        service.search_document(
            document_id="",
            query="Python",
        )

    except ValueError as error:
        assert "document" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError."
    )


def test_search_document_returns_empty_for_missing_document() -> None:
    service, _ = create_service([])

    results = service.search_document(
        document_id="does-not-exist",
        query="Python",
    )

    assert results == []
