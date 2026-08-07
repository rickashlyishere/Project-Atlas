from __future__ import annotations

from infrastructure.database.database import Database


class SchemaManager:
    """
    Creates Atlas database schema.
    """

    def __init__(self, database: Database) -> None:
        self.database = database

    def initialize(self) -> None:

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