from __future__ import annotations

from sentence_transformers import SentenceTransformer

from domain.embeddings import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Local embedding provider backed by Sentence Transformers.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """
        Return the dimensionality of the embedding model.
        """

        return self._model.get_sentence_embedding_dimension()

    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a single text.
        """

        if not text.strip():
            raise ValueError(
                "Cannot generate an embedding for empty text."
            )

        vector = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return vector.tolist()

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        """

        if not texts:
            return []

        if any(
            not text.strip()
            for text in texts
        ):
            raise ValueError(
                "Cannot generate embeddings for empty text."
            )

        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return [
            vector.tolist()
            for vector in vectors
        ]