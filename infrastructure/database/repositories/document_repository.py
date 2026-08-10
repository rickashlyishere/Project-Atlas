from __future__ import annotations

import hashlib
from pathlib import Path

from domain.document import Document

from infrastructure.database.database import Database


class DocumentRepository:
    """
    Repository responsible for document persistence.
    """

    def __init__(
        self,
        database: Database,
    ) -> None:
        self.database = database

    def add(
        self,
        document: Document,
    ) -> None:
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
                content_hash,
                created_at
            )
            VALUES
            (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
                document.content_hash,
                document.created_at.isoformat(),
            ),
        )

    def get_by_content_hash(
        self,
        content_hash: str,
    ):
        """
        Return the first document matching a SHA-256
        content hash.
        """

        content_hash = (
            content_hash.strip().lower()
        )

        if not content_hash:
            return None

        cursor = self.database.execute(
            """
            SELECT *
            FROM documents
            WHERE content_hash = ?
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (content_hash,),
        )

        return cursor.fetchone()

    def get_by_id(
        self,
        document_id: str,
    ):
        """
        Return a document row by ID.
        """

        document_id = document_id.strip()

        if not document_id:
            return None

        cursor = self.database.execute(
            """
            SELECT *
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        )

        return cursor.fetchone()

    def delete(
        self,
        document_id: str,
    ) -> bool:
        """
        Delete a document from the Atlas database.

        Associated chunks and embeddings are removed by
        the database foreign-key cascade.

        The original source file is not deleted.

        Returns True when a document was deleted.
        Returns False when the document did not exist.
        """

        document_id = document_id.strip()

        if not document_id:
            return False

        cursor = self.database.execute(
            """
            DELETE FROM documents
            WHERE id = ?
            """,
            (document_id,),
        )

        return cursor.rowcount > 0

    def backfill_content_hashes(self) -> int:
        """
        Calculate hashes for existing documents that do
        not yet have one.

        Returns the number of documents successfully updated.
        """

        rows = self.database.execute(
            """
            SELECT id, filepath
            FROM documents
            WHERE content_hash IS NULL
               OR content_hash = ''
            ORDER BY created_at ASC
            """
        ).fetchall()

        updated = 0

        for row in rows:
            filepath = Path(
                str(row["filepath"])
            )

            if not filepath.is_file():
                continue

            content_hash = (
                self._calculate_file_hash(
                    filepath
                )
            )

            self.database.execute(
                """
                UPDATE documents
                SET content_hash = ?
                WHERE id = ?
                """,
                (
                    content_hash,
                    row["id"],
                ),
            )

            updated += 1

        return updated

    @staticmethod
    def _calculate_file_hash(
        file_path: Path,
    ) -> str:
        """
        Calculate SHA-256 for a file.
        """

        digest = hashlib.sha256()

        with file_path.open("rb") as file:
            for block in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(block)

        return digest.hexdigest()

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