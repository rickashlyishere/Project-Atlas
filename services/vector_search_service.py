from __future__ import annotations

import math


class VectorSearchService:
    """
    Performs cosine-similarity search over persisted vectors.

    This is intentionally implemented without a vector database
    dependency. It gives Atlas a correct reference implementation
    before we introduce a specialized vector store.
    """

    def search(
        self,
        query_vector: list[float],
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Return the top-k most similar vectors.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if not query_vector:
            raise ValueError(
                "query_vector cannot be empty."
            )

        scored: list[dict] = []

        for candidate in candidates:
            vector = candidate["vector"]

            if len(vector) != len(query_vector):
                raise ValueError(
                    "Query and candidate vectors must "
                    "have the same dimension."
                )

            score = self.cosine_similarity(
                query_vector,
                vector,
            )

            scored.append(
                {
                    **candidate,
                    "score": score,
                }
            )

        scored.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return scored[:top_k]

    @staticmethod
    def cosine_similarity(
        first: list[float],
        second: list[float],
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        """

        if len(first) != len(second):
            raise ValueError(
                "Vectors must have the same dimension."
            )

        if not first:
            raise ValueError(
                "Vectors cannot be empty."
            )

        dot_product = sum(
            left * right
            for left, right in zip(first, second)
        )

        first_norm = math.sqrt(
            sum(value * value for value in first)
        )

        second_norm = math.sqrt(
            sum(value * value for value in second)
        )

        if first_norm == 0 or second_norm == 0:
            return 0.0

        return dot_product / (
            first_norm * second_norm
        )