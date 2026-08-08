from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """
    Domain-level abstraction for generating embeddings.

    The domain does not know which embedding model or framework
    is being used.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the dimensionality of generated vectors.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for one piece of text.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple pieces of text.
        """
        raise NotImplementedError