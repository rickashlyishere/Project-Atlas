from infrastructure.database import (
    Database,
    EmbeddingRepository,
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

    database.execute(
        """
        INSERT INTO chunks
        (
            id,
            document_id,
            text,
            page_number,
            chunk_type,
            embedding_id,
            created_at
        )
        VALUES
        (
            ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            "chunk-1",
            "document-1",
            "Atlas test chunk.",
            1,
            "fixed",
            None,
            "2026-01-01T00:00:00",
        ),
    )

    return database


def test_add_and_get_embedding() -> None:
    database = create_database()

    repository = EmbeddingRepository(database)

    repository.add(
        embedding_id="embedding-1",
        chunk_id="chunk-1",
        model_name="test-model",
        vector=[
            1.0,
            0.0,
            0.5,
        ],
    )

    result = repository.get_by_id(
        "embedding-1"
    )

    assert result is not None
    assert result["id"] == "embedding-1"
    assert result["chunk_id"] == "chunk-1"
    assert result["model_name"] == "test-model"
    assert result["dimension"] == 3
    assert result["vector"] == [
        1.0,
        0.0,
        0.5,
    ]


def test_get_embedding_by_chunk() -> None:
    database = create_database()

    repository = EmbeddingRepository(database)

    repository.add(
        embedding_id="embedding-1",
        chunk_id="chunk-1",
        model_name="test-model",
        vector=[1.0, 0.0, 0.0],
    )

    result = repository.get_by_chunk(
        "chunk-1"
    )

    assert result is not None
    assert result["id"] == "embedding-1"


def test_add_many_embeddings() -> None:
    database = create_database()

    database.execute(
        """
        INSERT INTO chunks
        (
            id,
            document_id,
            text,
            page_number,
            chunk_type,
            embedding_id,
            created_at
        )
        VALUES
        (
            ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            "chunk-2",
            "document-1",
            "Second chunk.",
            1,
            "fixed",
            None,
            "2026-01-01T00:00:00",
        ),
    )

    repository = EmbeddingRepository(database)

    repository.add_many(
        [
            {
                "id": "embedding-1",
                "chunk_id": "chunk-1",
                "model_name": "test-model",
                "vector": [1.0, 0.0],
            },
            {
                "id": "embedding-2",
                "chunk_id": "chunk-2",
                "model_name": "test-model",
                "vector": [0.0, 1.0],
            },
        ]
    )

    results = repository.get_all()

    assert len(results) == 2


def test_missing_embedding_returns_none() -> None:
    database = create_database()

    repository = EmbeddingRepository(database)

    assert (
        repository.get_by_id("missing")
        is None
    )


def test_delete_embedding_by_chunk() -> None:
    database = create_database()

    repository = EmbeddingRepository(database)

    repository.add(
        embedding_id="embedding-1",
        chunk_id="chunk-1",
        model_name="test-model",
        vector=[1.0, 0.0],
    )

    repository.delete_by_chunk(
        "chunk-1"
    )

    assert (
        repository.get_by_chunk("chunk-1")
        is None
    )