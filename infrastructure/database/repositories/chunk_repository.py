from __future__ import annotations

from datetime import datetime

from domain.document import Chunk, ChunkType

from infrastructure.database.database import Database


class ChunkRepository:
    """
    Repository responsible for chunk persistence.
    """

    def __init__(
        self,
        database: Database,
    ) -> None:
        self.database = database

    def add(
        self,
        document_id: str,
        chunk: Chunk,
    ) -> None:
        """
        Persist a single chunk.

        Existing chunks are updated in place instead of using
        INSERT OR REPLACE. This prevents foreign-key cascades
        from deleting associated embeddings.
        """

        self.database.execute(
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
            ON CONFLICT(id) DO UPDATE SET
                document_id = excluded.document_id,
                text = excluded.text,
                page_number = excluded.page_number,
                chunk_type = excluded.chunk_type,
                embedding_id = excluded.embedding_id
            """,
            (
                chunk.id,
                document_id,
                chunk.text,
                chunk.page_number,
                chunk.chunk_type.value,
                chunk.embedding_id,
                datetime.now().isoformat(),
            ),
        )

    def add_many(
        self,
        document_id: str,
        chunks: list[Chunk],
    ) -> None:
        """
        Persist multiple chunks in one transaction.

        Existing chunks are updated in place rather than
        replaced.
        """

        if not chunks:
            return

        self.database.executemany(
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
            ON CONFLICT(id) DO UPDATE SET
                document_id = excluded.document_id,
                text = excluded.text,
                page_number = excluded.page_number,
                chunk_type = excluded.chunk_type,
                embedding_id = excluded.embedding_id
            """,
            [
                (
                    chunk.id,
                    document_id,
                    chunk.text,
                    chunk.page_number,
                    chunk.chunk_type.value,
                    chunk.embedding_id,
                    datetime.now().isoformat(),
                )
                for chunk in chunks
            ],
        )

    def get_by_id(
        self,
        chunk_id: str,
    ) -> Chunk | None:
        """
        Retrieve one chunk by ID.
        """

        cursor = self.database.execute(
            """
            SELECT *
            FROM chunks
            WHERE id = ?
            """,
            (chunk_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_chunk(row)

    def get_by_document(
        self,
        document_id: str,
    ) -> list[Chunk]:
        """
        Retrieve all chunks belonging to a document.
        """

        cursor = self.database.execute(
            """
            SELECT *
            FROM chunks
            WHERE document_id = ?
            ORDER BY page_number, created_at
            """,
            (document_id,),
        )

        return [
            self._row_to_chunk(row)
            for row in cursor.fetchall()
        ]

    def delete_by_document(
        self,
        document_id: str,
    ) -> None:
        """
        Delete every chunk belonging to a document.
        """

        self.database.execute(
            """
            DELETE FROM chunks
            WHERE document_id = ?
            """,
            (document_id,),
        )

    def count_by_document(
        self,
        document_id: str,
    ) -> int:
        """
        Return the number of chunks belonging to a document.
        """

        cursor = self.database.execute(
            """
            SELECT COUNT(*) AS count
            FROM chunks
            WHERE document_id = ?
            """,
            (document_id,),
        )

        row = cursor.fetchone()

        return int(
            row["count"]
        )

    @staticmethod
    def _row_to_chunk(
        row,
    ) -> Chunk:
        """
        Convert a SQLite row into a domain Chunk.
        """

        return Chunk(
            id=row["id"],
            text=row["text"],
            page_number=row["page_number"],
            chunk_type=ChunkType(
                row["chunk_type"]
            ),
            embedding_id=row["embedding_id"],
        )