from domain.document import Chunk, ChunkType

from infrastructure.database import (
    ChunkRepository,
    Database,
    SchemaManager,
)


def create_database() -> Database:
    database = Database(":memory:")

    SchemaManager(database).initialize()

    database.execute(
        """
        INSERT INTO documents
        (
            id,
            filename,
            filepath,
            document_type,
            title,
            author,
            subject,
            page_count,
            file_size,
            created_at
        )
        VALUES
        (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            "document-1",
            "test.txt",
            "test.txt",
            "txt",
            "",
            "",
            "",
            1,
            100,
            "2026-01-01T00:00:00",
        ),
    )

    return database


def create_chunk(
    chunk_id: str = "chunk-1",
    page_number: int = 1,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text="Atlas chunk text.",
        page_number=page_number,
        chunk_type=ChunkType.FIXED,
    )


def test_add_and_get_chunk() -> None:
    database = create_database()

    repository = ChunkRepository(database)

    chunk = create_chunk()

    repository.add(
        document_id="document-1",
        chunk=chunk,
    )

    result = repository.get_by_id("chunk-1")

    assert result is not None
    assert result.id == "chunk-1"
    assert result.text == "Atlas chunk text."
    assert result.page_number == 1
    assert result.chunk_type == ChunkType.FIXED


def test_add_many_chunks() -> None:
    database = create_database()

    repository = ChunkRepository(database)

    chunks = [
        create_chunk("chunk-1", 1),
        create_chunk("chunk-2", 1),
        create_chunk("chunk-3", 2),
    ]

    repository.add_many(
        document_id="document-1",
        chunks=chunks,
    )

    result = repository.get_by_document(
        "document-1"
    )

    assert len(result) == 3
    assert result[0].id == "chunk-1"
    assert result[1].id == "chunk-2"
    assert result[2].id == "chunk-3"


def test_count_chunks() -> None:
    database = create_database()

    repository = ChunkRepository(database)

    repository.add_many(
        document_id="document-1",
        chunks=[
            create_chunk("chunk-1"),
            create_chunk("chunk-2"),
        ],
    )

    assert (
        repository.count_by_document("document-1")
        == 2
    )


def test_delete_document_chunks() -> None:
    database = create_database()

    repository = ChunkRepository(database)

    repository.add_many(
        document_id="document-1",
        chunks=[
            create_chunk("chunk-1"),
            create_chunk("chunk-2"),
        ],
    )

    repository.delete_by_document(
        "document-1"
    )

    assert (
        repository.count_by_document("document-1")
        == 0
    )


def test_missing_chunk_returns_none() -> None:
    database = create_database()

    repository = ChunkRepository(database)

    assert (
        repository.get_by_id("does-not-exist")
        is None
    )


def test_empty_add_many_does_nothing() -> None:
    database = create_database()

    repository = ChunkRepository(database)

    repository.add_many(
        document_id="document-1",
        chunks=[],
    )

    assert (
        repository.count_by_document("document-1")
        == 0
    )