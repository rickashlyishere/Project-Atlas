from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from infrastructure.database.database import Database


class EmbeddingRepository:
    """
    Repository responsible for persisting and retrieving embeddings.
    """

    def __init__(
        self,
        database: Database,
    ) -> None:
        self.database = database

    def add(
        self,
        embedding_id: str,
        chunk_id: str,
        model_name: str,
        vector: list[float],
    ) -> None:
        """
        Persist one embedding.
        """

        self.database.execute(
            """
            INSERT OR REPLACE INTO embeddings
            (
                id,
                chunk_id,
                model_name,
                dimension,
                vector,
                created_at
            )
            VALUES
            (
                ?, ?, ?, ?, ?, ?
            )
            """,
            (
                embedding_id,
                chunk_id,
                model_name,
                len(vector),
                json.dumps(vector),
                datetime.now().isoformat(),
            ),
        )

    def add_many(
        self,
        embeddings: list[dict[str, Any]],
    ) -> None:
        """
        Persist multiple embeddings.
        """

        if not embeddings:
            return

        self.database.executemany(
            """
            INSERT OR REPLACE INTO embeddings
            (
                id,
                chunk_id,
                model_name,
                dimension,
                vector,
                created_at
            )
            VALUES
            (
                ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    embedding["id"],
                    embedding["chunk_id"],
                    embedding["model_name"],
                    len(embedding["vector"]),
                    json.dumps(
                        embedding["vector"]
                    ),
                    datetime.now().isoformat(),
                )
                for embedding in embeddings
            ],
        )

    def get_by_id(
        self,
        embedding_id: str,
    ) -> dict[str, Any] | None:
        """
        Retrieve an embedding by ID.
        """

        cursor = self.database.execute(
            """
            SELECT *
            FROM embeddings
            WHERE id = ?
            """,
            (embedding_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def get_by_chunk(
        self,
        chunk_id: str,
    ) -> dict[str, Any] | None:
        """
        Retrieve the embedding belonging to a chunk.
        """

        cursor = self.database.execute(
            """
            SELECT *
            FROM embeddings
            WHERE chunk_id = ?
            """,
            (chunk_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def get_all(
        self,
    ) -> list[dict[str, Any]]:
        """
        Retrieve all embeddings.
        """

        cursor = self.database.execute(
            """
            SELECT *
            FROM embeddings
            ORDER BY created_at
            """
        )

        return [
            self._row_to_dict(row)
            for row in cursor.fetchall()
        ]

    def get_by_document(
        self,
        document_id: str,
    ) -> list[dict[str, Any]]:
        """
        Retrieve all embeddings belonging to a document.

        The returned records also contain chunk metadata needed
        by semantic search.
        """

        cursor = self.database.execute(
            """
            SELECT
                embeddings.id AS embedding_id,
                embeddings.chunk_id AS chunk_id,
                embeddings.model_name AS model_name,
                embeddings.dimension AS dimension,
                embeddings.vector AS vector,
                embeddings.created_at AS embedding_created_at,

                chunks.text AS text,
                chunks.page_number AS page_number,
                chunks.chunk_type AS chunk_type,

                documents.id AS document_id,
                documents.filename AS filename

            FROM embeddings

            INNER JOIN chunks
                ON embeddings.chunk_id = chunks.id

            INNER JOIN documents
                ON chunks.document_id = documents.id

            WHERE documents.id = ?

            ORDER BY
                chunks.page_number,
                embeddings.created_at
            """,
            (document_id,),
        )

        return [
            self._search_row_to_dict(row)
            for row in cursor.fetchall()
        ]

    def get_all_for_search(
        self,
    ) -> list[dict[str, Any]]:
        """
        Retrieve every embedding with its chunk metadata.
        """

        cursor = self.database.execute(
            """
            SELECT
                embeddings.id AS embedding_id,
                embeddings.chunk_id AS chunk_id,
                embeddings.model_name AS model_name,
                embeddings.dimension AS dimension,
                embeddings.vector AS vector,
                embeddings.created_at AS embedding_created_at,

                chunks.text AS text,
                chunks.page_number AS page_number,
                chunks.chunk_type AS chunk_type,

                documents.id AS document_id,
                documents.filename AS filename

            FROM embeddings

            INNER JOIN chunks
                ON embeddings.chunk_id = chunks.id

            INNER JOIN documents
                ON chunks.document_id = documents.id

            ORDER BY embeddings.created_at
            """
        )

        return [
            self._search_row_to_dict(row)
            for row in cursor.fetchall()
        ]

    def delete_by_chunk(
        self,
        chunk_id: str,
    ) -> None:
        """
        Delete the embedding associated with a chunk.
        """

        self.database.execute(
            """
            DELETE FROM embeddings
            WHERE chunk_id = ?
            """,
            (chunk_id,),
        )

    @staticmethod
    def _row_to_dict(row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "chunk_id": row["chunk_id"],
            "model_name": row["model_name"],
            "dimension": row["dimension"],
            "vector": json.loads(
                row["vector"]
            ),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _search_row_to_dict(row) -> dict[str, Any]:
        return {
            "embedding_id": row["embedding_id"],
            "chunk_id": row["chunk_id"],
            "model_name": row["model_name"],
            "dimension": row["dimension"],
            "vector": json.loads(
                row["vector"]
            ),
            "created_at": row[
                "embedding_created_at"
            ],
            "text": row["text"],
            "page_number": row["page_number"],
            "chunk_type": row["chunk_type"],
            "document_id": row["document_id"],
            "filename": row["filename"],
        }