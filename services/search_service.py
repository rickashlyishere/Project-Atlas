from __future__ import annotations

from typing import Any, Protocol


class EmbeddingServiceProtocol(Protocol):
    """
    Interface required by SearchService for generating
    query embeddings.
    """

    @property
    def dimension(self) -> int:
        ...

    @property
    def provider(self) -> Any:
        ...


class EmbeddingRepositoryProtocol(Protocol):
    """
    Interface required by SearchService for retrieving
    persisted embeddings.
    """

    def get_all_for_search(
        self,
    ) -> list[dict[str, Any]]:
        ...

    def get_by_document(
        self,
        document_id: str,
    ) -> list[dict[str, Any]]:
        ...


class VectorSearchServiceProtocol(Protocol):
    """
    Interface required by SearchService for vector comparison.
    """

    def search(
        self,
        query_vector: list[float],
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        ...


class SearchService:
    """
    High-level semantic search service.

    Converts a natural-language query into an embedding,
    compares it with stored chunk embeddings, and returns
    ranked chunks.
    """

    def __init__(
        self,
        embedding_service: EmbeddingServiceProtocol,
        embedding_repository: EmbeddingRepositoryProtocol,
        vector_search_service: VectorSearchServiceProtocol,
    ) -> None:
        self.embedding_service = embedding_service
        self.embedding_repository = embedding_repository
        self.vector_search_service = vector_search_service

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic search across all indexed chunks.
        """

        query = query.strip()

        if not query:
            raise ValueError(
                "Search query cannot be empty."
            )

        query_vector = (
            self.embedding_service.provider.embed_text(
                query
            )
        )

        candidates = (
            self.embedding_repository
            .get_all_for_search()
        )

        if not candidates:
            return []

        return self.vector_search_service.search(
            query_vector=query_vector,
            candidates=candidates,
            top_k=top_k,
        )

    def search_document(
        self,
        document_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic search within one document.
        """

        query = query.strip()

        if not query:
            raise ValueError(
                "Search query cannot be empty."
            )

        query_vector = (
            self.embedding_service.provider.embed_text(
                query
            )
        )

        candidates = (
            self.embedding_repository
            .get_by_document(
                document_id
            )
        )

        if not candidates:
            return []

        return self.vector_search_service.search(
            query_vector=query_vector,
            candidates=candidates,
            top_k=top_k,
        )