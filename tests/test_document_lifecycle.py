from __future__ import annotations

from pathlib import Path

import pytest

from domain.embeddings import EmbeddingProvider

from infrastructure.database import Database

from services.document_service import DocumentService
from services.embedding_service import EmbeddingService


class FakeEmbeddingProvider(
    EmbeddingProvider
):
    """
    Deterministic embedding provider for lifecycle tests.

    No machine-learning model is loaded.
    """

    model_name = "test-model"

    @property
    def dimension(self) -> int:
        return 3

    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        if not text.strip():
            raise ValueError(
                "Cannot embed empty text."
            )

        return [
            1.0,
            0.0,
            0.0,
        ]

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if any(
            not text.strip()
            for text in texts
        ):
            raise ValueError(
                "Cannot embed empty text."
            )

        return [
            [
                1.0,
                0.0,
                0.0,
            ]
            for _ in texts
        ]


def create_service() -> DocumentService:
    """
    Create a DocumentService using an in-memory
    SQLite database and deterministic embeddings.
    """

    database = Database(
        ":memory:"
    )

    base_service = DocumentService(
        database=database
    )

    provider = FakeEmbeddingProvider()

    embedding_service = EmbeddingService(
        provider=provider,
        repository=(
            base_service.embedding_repository
        ),
    )

    return DocumentService(
        database=database,
        embedding_service=embedding_service,
    )


def create_text_file(
    tmp_path: Path,
    filename: str,
    content: str,
) -> Path:
    """
    Create a temporary text document.
    """

    file_path = (
        tmp_path / filename
    )

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    return file_path


def test_delete_document_removes_document(
    tmp_path: Path,
) -> None:
    """
    Deleting a document removes it from the
    document repository.
    """

    file_path = create_text_file(
        tmp_path,
        "delete.txt",
        "Atlas deletion test.",
    )

    service = create_service()

    document = service.load(
        file_path
    )

    assert len(
        service.list_documents()
    ) == 1

    deleted = service.delete(
        document.id
    )

    assert deleted is True

    assert len(
        service.list_documents()
    ) == 0


def test_delete_document_removes_chunks(
    tmp_path: Path,
) -> None:
    """
    Deleting a document removes all of its chunks.
    """

    file_path = create_text_file(
        tmp_path,
        "chunks.txt",
        (
            "Atlas chunk deletion test. "
            "This document contains content "
            "that will be chunked and persisted."
        ),
    )

    service = create_service()

    document = service.load(
        file_path
    )

    chunks_before = service.get_chunks(
        document.id
    )

    assert len(chunks_before) > 0

    deleted = service.delete(
        document.id
    )

    assert deleted is True

    chunks_after = service.get_chunks(
        document.id
    )

    assert chunks_after == []


def test_delete_document_removes_embeddings(
    tmp_path: Path,
) -> None:
    """
    Deleting a document removes embeddings associated
    with its chunks through the database cascade.
    """

    file_path = create_text_file(
        tmp_path,
        "embeddings.txt",
        "Atlas embedding deletion test.",
    )

    service = create_service()

    document = service.load(
        file_path
    )

    embeddings_before = (
        service.embedding_repository
        .get_by_document(
            document.id
        )
    )

    assert len(embeddings_before) > 0

    deleted = service.delete(
        document.id
    )

    assert deleted is True

    embeddings_after = (
        service.embedding_repository
        .get_by_document(
            document.id
        )
    )

    assert embeddings_after == []


def test_delete_document_removes_chunks_and_embeddings_together(
    tmp_path: Path,
) -> None:
    """
    Document deletion removes the complete indexed
    representation of that document.
    """

    file_path = create_text_file(
        tmp_path,
        "complete_delete.txt",
        (
            "Atlas stores documents, chunks, "
            "and embeddings."
        ),
    )

    service = create_service()

    document = service.load(
        file_path
    )

    chunks = service.get_chunks(
        document.id
    )

    embeddings = (
        service.embedding_repository
        .get_by_document(
            document.id
        )
    )

    assert len(chunks) > 0
    assert len(embeddings) > 0

    assert service.delete(
        document.id
    ) is True

    assert (
        service.get_chunks(
            document.id
        )
        == []
    )

    assert (
        service.embedding_repository
        .get_by_document(
            document.id
        )
        == []
    )

    assert all(
        service.get_embedding(
            chunk.id
        )
        is None
        for chunk in chunks
    )


def test_delete_document_does_not_delete_source_file(
    tmp_path: Path,
) -> None:
    """
    Removing a document from Atlas must not delete
    the original source file.
    """

    file_path = create_text_file(
        tmp_path,
        "source.txt",
        "The original source must remain.",
    )

    service = create_service()

    document = service.load(
        file_path
    )

    assert file_path.exists()

    assert service.delete(
        document.id
    ) is True

    assert file_path.exists()

    assert (
        file_path.read_text(
            encoding="utf-8"
        )
        == "The original source must remain."
    )


def test_delete_nonexistent_document_returns_false() -> None:
    """
    Deleting a document ID that does not exist
    should not raise an exception.
    """

    service = create_service()

    deleted = service.delete(
        "does-not-exist"
    )

    assert deleted is False


def test_delete_empty_document_id_is_rejected() -> None:
    """
    Empty document IDs are invalid input.
    """

    service = create_service()

    with pytest.raises(
        ValueError,
        match="Document ID cannot be empty",
    ):
        service.delete(
            ""
        )


def test_delete_one_document_preserves_another(
    tmp_path: Path,
) -> None:
    """
    Deleting one document must not affect another
    document or its indexed data.
    """

    first_file = create_text_file(
        tmp_path,
        "first.txt",
        "First Atlas document.",
    )

    second_file = create_text_file(
        tmp_path,
        "second.txt",
        "Second Atlas document.",
    )

    service = create_service()

    first = service.load(
        first_file
    )

    second = service.load(
        second_file
    )

    second_chunks_before = (
        service.get_chunks(
            second.id
        )
    )

    second_embeddings_before = (
        service.embedding_repository
        .get_by_document(
            second.id
        )
    )

    assert service.delete(
        first.id
    ) is True

    documents = (
        service.list_documents()
    )

    assert len(documents) == 1

    assert str(
        documents[0]["id"]
    ) == second.id

    second_chunks_after = (
        service.get_chunks(
            second.id
        )
    )

    second_embeddings_after = (
        service.embedding_repository
        .get_by_document(
            second.id
        )
    )

    assert len(second_chunks_after) == (
        len(second_chunks_before)
    )

    assert len(second_embeddings_after) == (
        len(second_embeddings_before)
    )

    assert (
        "Second Atlas document"
        in " ".join(
            chunk.text
            for chunk in second_chunks_after
        )
    )