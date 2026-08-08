from domain.document import (
    ChunkType,
    Document,
    DocumentType,
    Page,
)
from services.chunk_service import ChunkService


def create_document(text: str) -> Document:
    return Document(
        id="test-document",
        filename="test.txt",
        filepath="test.txt",
        document_type=DocumentType.TXT,
        pages=[
            Page(
                number=1,
                text=text,
            )
        ],
        extracted_text=text,
        page_count=1,
    )


def test_fixed_chunk_document_creates_chunks() -> None:
    document = create_document(
        "A" * 2500
    )

    service = ChunkService(
        strategy=ChunkType.FIXED,
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = service.chunk_document(document)

    assert len(chunks) == 3

    assert all(
        chunk.chunk_type == ChunkType.FIXED
        for chunk in chunks
    )


def test_fixed_chunk_overlap_is_preserved() -> None:
    text = "".join(
        str(index % 10)
        for index in range(1500)
    )

    document = create_document(text)

    service = ChunkService(
        strategy=ChunkType.FIXED,
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = service.chunk_document(document)

    assert len(chunks) == 2

    assert chunks[0].text[-200:] == chunks[1].text[:200]


def test_page_chunks_are_attached_to_page() -> None:
    document = create_document(
        "Atlas document processing " * 100
    )

    service = ChunkService(
        strategy=ChunkType.FIXED,
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = service.chunk_document(document)

    assert document.pages[0].chunks == chunks


def test_empty_page_creates_no_chunks() -> None:
    document = create_document("")

    service = ChunkService()

    chunks = service.chunk_document(document)

    assert chunks == []
    assert document.pages[0].chunks == []


def test_invalid_overlap_is_rejected() -> None:
    try:
        ChunkService(
            chunk_size=100,
            chunk_overlap=100,
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for invalid overlap."
    )


def test_paragraph_chunking_preserves_paragraphs() -> None:
    text = (
        "Paragraph one contains information about Atlas.\n\n"
        "Paragraph two contains information about documents.\n\n"
        "Paragraph three contains information about search."
    )

    service = ChunkService(
        strategy=ChunkType.PARAGRAPH,
        chunk_size=500,
        chunk_overlap=0,
    )

    chunks = service.chunk_page(
        page_number=1,
        text=text,
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_type == ChunkType.PARAGRAPH
    assert "Paragraph one" in chunks[0].text
    assert "Paragraph two" in chunks[0].text
    assert "Paragraph three" in chunks[0].text


def test_paragraph_chunking_creates_multiple_chunks() -> None:
    text = (
        "First paragraph. " * 10
        + "\n\n"
        + "Second paragraph. " * 10
        + "\n\n"
        + "Third paragraph. " * 10
    )

    service = ChunkService(
        strategy=ChunkType.PARAGRAPH,
        chunk_size=150,
        chunk_overlap=0,
    )

    chunks = service.chunk_page(
        page_number=1,
        text=text,
    )

    assert len(chunks) >= 2

    assert all(
        chunk.chunk_type == ChunkType.PARAGRAPH
        for chunk in chunks
    )


def test_sentence_chunking_preserves_sentence_boundaries() -> None:
    text = (
        "Atlas reads documents. "
        "Documents are converted into text. "
        "Text can be divided into chunks."
    )

    service = ChunkService(
        strategy=ChunkType.SENTENCE,
        chunk_size=1000,
        chunk_overlap=0,
    )

    chunks = service.chunk_page(
        page_number=1,
        text=text,
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_type == ChunkType.SENTENCE
    assert chunks[0].text.endswith("chunks.")


def test_sentence_chunking_creates_multiple_chunks() -> None:
    text = " ".join(
        f"Sentence number {index}."
        for index in range(30)
    )

    service = ChunkService(
        strategy=ChunkType.SENTENCE,
        chunk_size=100,
        chunk_overlap=0,
    )

    chunks = service.chunk_page(
        page_number=1,
        text=text,
    )

    assert len(chunks) > 1

    assert all(
        chunk.chunk_type == ChunkType.SENTENCE
        for chunk in chunks
    )


def test_semantic_strategy_is_not_available_yet() -> None:
    try:
        ChunkService(
            strategy=ChunkType.SEMANTIC,
        )
    except ValueError as error:
        assert "not implemented" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError for semantic strategy."
    )