from pathlib import Path

from domain.document import DocumentType
from domain.embeddings import EmbeddingProvider

from infrastructure.database import Database

from services.document_service import DocumentService
from services.embedding_service import EmbeddingService


class FakeEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic embedding provider for integration tests.

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
    Create a DocumentService using:

    - an in-memory SQLite database
    - a fake embedding provider
    - the real application services
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
        repository=base_service.embedding_repository,
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

    file_path = tmp_path / filename

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    return file_path


def test_document_ingestion_creates_persisted_chunks(
    tmp_path: Path,
) -> None:

    file_path = create_text_file(
        tmp_path,
        "atlas_test.txt",
        (
            "Atlas is a document processing platform.\n\n"
            "It extracts text from documents and prepares "
            "that text for downstream AI processing.\n\n"
            "The chunking layer divides documents into smaller "
            "pieces that can later be embedded and searched."
        ),
    )

    service = create_service()

    document = service.load(
        file_path
    )

    assert document is not None

    assert document.filename == "atlas_test.txt"

    assert (
        document.document_type
        == DocumentType.TXT
    )

    chunks = service.get_chunks(
        document.id
    )

    assert len(chunks) > 0

    assert all(
        chunk.page_number >= 1
        for chunk in chunks
    )

    assert all(
        chunk.text.strip()
        for chunk in chunks
    )


def test_ingested_chunks_belong_to_correct_document(
    tmp_path: Path,
) -> None:

    first_file = create_text_file(
        tmp_path,
        "first.txt",
        "This is the first Atlas document.",
    )

    second_file = create_text_file(
        tmp_path,
        "second.txt",
        "This is the second Atlas document.",
    )

    service = create_service()

    first_document = service.load(
        first_file
    )

    second_document = service.load(
        second_file
    )

    first_chunks = service.get_chunks(
        first_document.id
    )

    second_chunks = service.get_chunks(
        second_document.id
    )

    assert len(first_chunks) > 0
    assert len(second_chunks) > 0

    first_text = " ".join(
        chunk.text
        for chunk in first_chunks
    )

    second_text = " ".join(
        chunk.text
        for chunk in second_chunks
    )

    assert (
        "first Atlas document"
        in first_text
    )

    assert (
        "second Atlas document"
        in second_text
    )

    assert (
        "second Atlas document"
        not in first_text
    )

    assert (
        "first Atlas document"
        not in second_text
    )


def test_chunk_page_numbers_are_preserved(
    tmp_path: Path,
) -> None:

    file_path = create_text_file(
        tmp_path,
        "page.txt",
        "Atlas page content.",
    )

    service = create_service()

    document = service.load(
        file_path
    )

    chunks = service.get_chunks(
        document.id
    )

    assert len(chunks) > 0

    assert all(
        chunk.page_number == 1
        for chunk in chunks
    )


def test_ingested_chunks_have_embeddings(
    tmp_path: Path,
) -> None:

    file_path = create_text_file(
        tmp_path,
        "embedding.txt",
        "Atlas is generating test embeddings.",
    )

    service = create_service()

    document = service.load(
        file_path
    )

    chunks = service.get_chunks(
        document.id
    )

    assert len(chunks) > 0

    assert all(
        chunk.embedding_id is not None
        for chunk in chunks
    )

    for chunk in chunks:

        embedding = service.get_embedding(
            chunk.id
        )

        assert embedding is not None

        assert (
            embedding["chunk_id"]
            == chunk.id
        )

        assert (
            embedding["dimension"]
            == 3
        )

        assert len(
            embedding["vector"]
        ) == 3

        assert (
            embedding["model_name"]
            == "test-model"
        )