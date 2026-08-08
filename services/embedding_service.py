from __future__ import annotations

from domain.document import Chunk
from domain.embeddings import EmbeddingProvider


class EmbeddingService:
    """
    Application service responsible for generating embeddings
    for document chunks.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
    ) -> None:
        self.provider = provider

    @property
    def dimension(self) -> int:
        """
        Return the embedding dimension.
        """

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