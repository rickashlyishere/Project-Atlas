from __future__ import annotations

from infrastructure.database.database import Database


class SchemaManager:
    """
    Creates and maintains the Atlas database schema.
    """

    def __init__(self, database: Database) -> None:
        self.database = database

    def initialize(self) -> None:
        self._create_documents_table()
        self._create_chunks_table()
        self._create_embeddings_table()

    def _create_documents_table(self) -> None:
        self.database.execute(
            """
            CREATE TABLE IF NOT EXISTS documents
            (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                document_type TEXT NOT NULL,

                title TEXT,
                author TEXT,
                subject TEXT,

                page_count INTEGER,
                file_size INTEGER,

                created_at TEXT
            );
            """
        )

    def _create_chunks_table(self) -> None:
        self.database.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks
            (
                id TEXT PRIMARY KEY,

                document_id TEXT NOT NULL,

                text TEXT NOT NULL,

                page_number INTEGER NOT NULL,

                chunk_type TEXT NOT NULL,

                embedding_id TEXT,

                created_at TEXT NOT NULL,

                FOREIGN KEY (document_id)
                    REFERENCES documents(id)
                    ON DELETE CASCADE
            );
            """
        )

        self.database.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_chunks_document_id
            ON chunks(document_id);
            """
        )

        self.database.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_chunks_page_number
            ON chunks(document_id, page_number);
            """
        )

    def _create_embeddings_table(self) -> None:
        self.database.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings
            (
                id TEXT PRIMARY KEY,

                chunk_id TEXT NOT NULL UNIQUE,

                model_name TEXT NOT NULL,

                dimension INTEGER NOT NULL,

                vector TEXT NOT NULL,

                created_at TEXT NOT NULL,

                FOREIGN KEY (chunk_id)
                    REFERENCES chunks(id)
                    ON DELETE CASCADE
            );
            """
        )

        self.database.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_embeddings_chunk_id
            ON embeddings(chunk_id);
            """
        )