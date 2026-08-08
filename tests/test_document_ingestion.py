from pathlib import Path

from domain.document import DocumentType
from services.document_service import DocumentService


def create_text_file(
    tmp_path: Path,
    content: str,
) -> Path:
    file_path = tmp_path / "atlas_test.txt"

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    return file_path


def test_document_ingestion_creates_persisted_chunks(
    tmp_path: Path,
) -> None:
    content = (
        "Atlas is a document processing platform.\n\n"
        "It extracts text from documents and prepares that "
        "text for downstream AI processing.\n\n"
        "The chunking layer divides documents into smaller "
        "pieces that can later be embedded and searched."
    )

    file_path = create_text_file(
        tmp_path,
        content,
    )

    service = DocumentService()

    document = service.load(file_path)

    assert document is not None
    assert document.filename == "atlas_test.txt"
    assert document.document_type == DocumentType.TXT

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
        "This is the first Atlas document.",
    )

    second_file = tmp_path / "second.txt"

    second_file.write_text(
        "This is the second Atlas document.",
        encoding="utf-8",
    )

    service = DocumentService()

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

    assert "first Atlas document" in first_text
    assert "second Atlas document" in second_text

    assert "second Atlas document" not in first_text
    assert "first Atlas document" not in second_text


def test_chunk_page_numbers_are_preserved(
    tmp_path: Path,
) -> None:
    file_path = create_text_file(
        tmp_path,
        "Atlas page content.",
    )

    service = DocumentService()

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


def test_ingested_chunks_have_no_embedding_yet(
    tmp_path: Path,
) -> None:
    file_path = create_text_file(
        tmp_path,
        "Atlas is not embedding this text yet.",
    )

    service = DocumentService()

    document = service.load(
        file_path
    )

    chunks = service.get_chunks(
        document.id
    )

    assert len(chunks) > 0

    assert all(
        chunk.embedding_id is None
        for chunk in chunks
    )