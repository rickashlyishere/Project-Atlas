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
    compares it with stored chunk embeddings, removes
    duplicate chunk content, and returns ranked chunks.
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

    @staticmethod
    def _normalize_text(
        text: Any,
    ) -> str:
        """
        Normalize chunk text for duplicate detection.

        Whitespace differences should not cause two otherwise
        identical chunks to be treated as different.
        """

        return " ".join(
            str(text).split()
        ).strip().lower()

    @classmethod
    def _deduplicate_candidates(
        cls,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Remove duplicate document chunks before vector search.

        The duplicate key uses:
            filename
            page number
            normalized text

        This allows multiple legitimate chunks from the same
        document while removing repeated copies of the same
        document/page content.
        """

        unique_candidates: list[
            dict[str, Any]
        ] = []

        seen: set[
            tuple[str, int, str]
        ] = set()

        for candidate in candidates:
            filename = str(
                candidate.get(
                    "filename",
                    "",
                )
            ).strip().lower()

            page_number = int(
                candidate.get(
                    "page_number",
                    0,
                )
            )

            text = cls._normalize_text(
                candidate.get(
                    "text",
                    "",
                )
            )

            key = (
                filename,
                page_number,
                text,
            )

            if key in seen:
                continue

            seen.add(key)

            unique_candidates.append(
                candidate
            )

        return unique_candidates

    @staticmethod
    def _deduplicate_results(
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Remove duplicate results while preserving ranking order.

        This is a defensive second layer in case the vector-search
        implementation or another caller supplies duplicate results.
        """

        unique_results: list[
            dict[str, Any]
        ] = []

        seen: set[
            tuple[str, int, str]
        ] = set()

        for result in results:
            filename = str(
                result.get(
                    "filename",
                    "",
                )
            ).strip().lower()

            page_number = int(
                result.get(
                    "page_number",
                    0,
                )
            )

            text = " ".join(
                str(
                    result.get(
                        "text",
                        "",
                    )
                ).split()
            ).strip().lower()

            key = (
                filename,
                page_number,
                text,
            )

            if key in seen:
                continue

            seen.add(key)

            unique_results.append(
                result
            )

        return unique_results

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

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
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

        candidates = (
            self._deduplicate_candidates(
                candidates
            )
        )

        if not candidates:
            return []

        # Ask the vector layer to rank all unique candidates.
        # We apply top_k after deduplication so duplicate records
        # cannot consume the requested result slots.
        ranked_results = (
            self.vector_search_service.search(
                query_vector=query_vector,
                candidates=candidates,
                top_k=len(candidates),
            )
        )

        unique_results = (
            self._deduplicate_results(
                ranked_results
            )
        )

        return unique_results[:top_k]

    def search_document(
        self,
        document_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic search within one document.
        """

        document_id = document_id.strip()
        query = query.strip()

        if not document_id:
            raise ValueError(
                "Document ID cannot be empty."
            )

        if not query:
            raise ValueError(
                "Search query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
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

        candidates = (
            self._deduplicate_candidates(
                candidates
            )
        )

        if not candidates:
            return []

        ranked_results = (
            self.vector_search_service.search(
                query_vector=query_vector,
                candidates=candidates,
                top_k=len(candidates),
            )
        )

        unique_results = (
            self._deduplicate_results(
                ranked_results
            )
        )

        return unique_results[:top_k]
