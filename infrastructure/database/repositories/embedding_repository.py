from __future__ import annotations

import json
from datetime import datetime

from infrastructure.database.database import Database


class EmbeddingRepository:
    """
    Repository responsible for persisting and retrieving embeddings.
    """

    def __init__(self, database: Database) -> None:
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
        embeddings: list[dict],
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
                    json.dumps(embedding["vector"]),
                    datetime.now().isoformat(),
                )
                for embedding in embeddings
            ],
        )

    def get_by_id(
        self,
        embedding_id: str,
    ) -> dict | None:
        """
        Retrieve an embedding by its ID.
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
    ) -> dict | None:
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

    def get_all(self) -> list[dict]:
        """
        Retrieve all persisted embeddings.
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
    def _row_to_dict(row) -> dict:
        return {
            "id": row["id"],
            "chunk_id": row["chunk_id"],
            "model_name": row["model_name"],
            "dimension": row["dimension"],
            "vector": json.loads(row["vector"]),
            "created_at": row["created_at"],
        }