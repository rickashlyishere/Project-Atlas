from __future__ import annotations

from typing import Protocol
from uuid import uuid4

from domain.document import Chunk
from domain.embeddings import EmbeddingProvider


class EmbeddingRepositoryProtocol(Protocol):
    """
    Interface required by EmbeddingService for persistence.

    Both the real EmbeddingRepository and test doubles can
    implement this interface without inheritance.
    """

    def add_many(
        self,
        embeddings: list[dict],
    ) -> None:
        ...


class EmbeddingService:
    """
    Application service responsible for generating and
    persisting embeddings for document chunks.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        repository: EmbeddingRepositoryProtocol,
    ) -> None:
        self.provider = provider
        self.repository = repository

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    def embed_chunk(
        self,
        chunk: Chunk,
    ) -> list[float]:
        """
        Generate an embedding for one chunk.
        """

        return self.provider.embed_text(
            chunk.text
        )

    def embed_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple chunks.
        """

        if not chunks:
            return []

        return self.provider.embed_texts(
            [
                chunk.text
                for chunk in chunks
            ]
        )

    def embed_and_persist(
        self,
        chunks: list[Chunk],
    ) -> list[dict]:
        """
        Generate embeddings and persist them.

        Returns the persisted embedding records.
        """

        if not chunks:
            return []

        vectors = self.embed_chunks(chunks)

        embeddings: list[dict] = []

        model_name = getattr(
            self.provider,
            "model_name",
            self.provider.__class__.__name__,
        )

        for chunk, vector in zip(
            chunks,
            vectors,
            strict=True,
        ):
            embedding_id = str(uuid4())

            embeddings.append(
                {
                    "id": embedding_id,
                    "chunk_id": chunk.id,
                    "model_name": model_name,
                    "vector": vector,
                }
            )

            chunk.embedding_id = embedding_id

        self.repository.add_many(
            embeddings
        )

        return embeddings