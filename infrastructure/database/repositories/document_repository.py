from __future__ import annotations

from domain.document import Document

from infrastructure.database.database import Database


class DocumentRepository:
    """
    Repository responsible for document persistence.
    """

    def __init__(self, database: Database) -> None:
        self.database = database

    def add(self, document: Document) -> None:
        """
        Insert a document into the database.
        """

        self.database.execute(
            """
            INSERT OR REPLACE INTO documents
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
                document.id,
                document.filename,
                str(document.filepath),
                document.document_type.value,
                document.metadata.title,
                document.metadata.author,
                document.metadata.subject,
                document.page_count,
                document.file_size,
                document.created_at.isoformat(),
            ),
        )

    def list_all(self) -> list:
        """
        Return all stored documents.
        """

        cursor = self.database.execute(
            """
            SELECT *
            FROM documents
            ORDER BY filename
            """
        )

        return cursor.fetchall()