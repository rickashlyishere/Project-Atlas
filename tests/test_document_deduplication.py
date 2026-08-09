from __future__ import annotations

from pathlib import Path

from domain.document import DocumentType
from domain.embeddings import EmbeddingProvider

from infrastructure.database import Database

from services.document_service import DocumentService
from services.embedding_service import EmbeddingService


class FakeEmbeddingProvider(
    EmbeddingProvider
):
    """
    Deterministic embedding provider for
    ingestion tests.
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

    file_path = (
        tmp_path / filename
    )

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    return file_path


def test_same_file_is_not_ingested_twice(
    tmp_path: Path,
) -> None:

    file_path = create_text_file(
        tmp_path,
        "atlas.txt",
        "Atlas is an offline AI knowledge platform.",
    )

    service = create_service()

    first = service.load(
        file_path
    )

    first_chunks = service.get_chunks(
        first.id
    )

    documents_after_first = (
        service.list_documents()
    )

    second = service.load(
        file_path
    )

    second_chunks = service.get_chunks(
        second.id
    )

    documents_after_second = (
        service.list_documents()
    )

    assert first.id == second.id

    assert first.content_hash

    assert (
        first.content_hash
        == second.content_hash
    )

    assert len(
        documents_after_first
    ) == 1

    assert len(
        documents_after_second
    ) == 1

    assert len(first_chunks) > 0

    assert (
        len(second_chunks)
        == len(first_chunks)
    )


def test_same_content_with_different_filename_is_not_duplicated(
    tmp_path: Path,
) -> None:

    first_file = create_text_file(
        tmp_path,
        "first.txt",
        "The exact same Atlas content.",
    )

    second_file = create_text_file(
        tmp_path,
        "second.txt",
        "The exact same Atlas content.",
    )

    service = create_service()

    first = service.load(
        first_file
    )

    second = service.load(
        second_file
    )

    documents = (
        service.list_documents()
    )

    assert first.id == second.id

    assert (
        first.content_hash
        == second.content_hash
    )

    assert len(documents) == 1


def test_same_filename_with_different_content_is_allowed(
    tmp_path: Path,
) -> None:

    # Create two different directories so that both files
    # can legitimately have the same filename without one
    # overwriting the other.
    first_directory = (
        tmp_path / "first"
    )

    second_directory = (
        tmp_path / "second"
    )

    first_directory.mkdir()

    second_directory.mkdir()

    first_file = create_text_file(
        first_directory,
        "notes.txt",
        "This is the first document.",
    )

    second_file = create_text_file(
        second_directory,
        "notes.txt",
        "This is the second document.",
    )

    service = create_service()

    first = service.load(
        first_file
    )

    second = service.load(
        second_file
    )

    documents = (
        service.list_documents()
    )

    assert first.id != second.id

    assert (
        first.content_hash
        != second.content_hash
    )

    assert len(documents) == 2

    first_chunks = service.get_chunks(
        first.id
    )

    second_chunks = service.get_chunks(
        second.id
    )

    first_text = " ".join(
        chunk.text
        for chunk in first_chunks
    )

    second_text = " ".join(
        chunk.text
        for chunk in second_chunks
    )

    assert (
        "first document"
        in first_text
    )

    assert (
        "second document"
        in second_text
    )


def test_content_hash_is_sha256_length(
    tmp_path: Path,
) -> None:

    file_path = create_text_file(
        tmp_path,
        "hash.txt",
        "Atlas hashing test.",
    )

    service = create_service()

    document = service.load(
        file_path
    )

    assert len(
        document.content_hash
    ) == 64

    assert all(
        character
        in "0123456789abcdef"
        for character in document.content_hash
    )


def test_duplicate_does_not_create_more_embeddings(
    tmp_path: Path,
) -> None:

    file_path = create_text_file(
        tmp_path,
        "embedding.txt",
        "Atlas should only embed this once.",
    )

    service = create_service()

    first = service.load(
        file_path
    )

    first_embedding_count = len(
        service.embedding_repository
        .get_by_document(
            first.id
        )
    )

    service.load(
        file_path
    )

    second_embedding_count = len(
        service.embedding_repository
        .get_by_document(
            first.id
        )
    )

    assert (
        second_embedding_count
        == first_embedding_count
    )


def test_duplicate_preserves_document_type(
    tmp_path: Path,
) -> None:

    file_path = create_text_file(
        tmp_path,
        "type.txt",
        "Atlas document type test.",
    )

    service = create_service()

    first = service.load(
        file_path
    )

    second = service.load(
        file_path
    )

    assert (
        first.document_type
        == DocumentType.TXT
    )

    assert (
        second.document_type
        == DocumentType.TXT
    )

    assert first.id == second.id
