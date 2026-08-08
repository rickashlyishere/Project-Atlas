from domain.document import Chunk, ChunkType
from domain.embeddings import EmbeddingProvider
from services.embedding_service import EmbeddingService


class FakeEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic provider used for unit testing.

    It does not load an actual ML model.
    """

    @property
    def dimension(self) -> int:
        return 3

    def embed_text(
        self,
        text: str,
    ) -> list[float]:

        if not text.strip():
            raise ValueError(
                "Cannot embed empty text."
            )

        return [
            1.0,
            0.0,
            0.0,
        ]

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if any(
            not text.strip()
            for text in texts
        ):
            raise ValueError(
                "Cannot embed empty text."
            )

        return [
            [
                1.0,
                0.0,
                0.0,
            ]
            for _ in texts
        ]


def create_chunk(
    text: str = "Atlas document chunk.",
) -> Chunk:

    return Chunk(
        id="chunk-1",
        text=text,
        page_number=1,
        chunk_type=ChunkType.FIXED,
    )


def test_embedding_service_returns_dimension() -> None:
    provider = FakeEmbeddingProvider()

    service = EmbeddingService(provider)

    assert service.dimension == 3


def test_embed_chunk() -> None:
    provider = FakeEmbeddingProvider()

    service = EmbeddingService(provider)

    vector = service.embed_chunk(
        create_chunk()
    )

    assert vector == [
        1.0,
        0.0,
        0.0,
    ]


def test_embed_chunks() -> None:
    provider = FakeEmbeddingProvider()

    service = EmbeddingService(provider)

    chunks = [
        create_chunk("First chunk."),
        create_chunk("Second chunk."),
        create_chunk("Third chunk."),
    ]

    vectors = service.embed_chunks(chunks)

    assert len(vectors) == 3

    assert all(
        len(vector) == 3
        for vector in vectors
    )


def test_empty_chunk_list_returns_empty_vectors() -> None:
    provider = FakeEmbeddingProvider()

    service = EmbeddingService(provider)

    assert service.embed_chunks([]) == []


def test_empty_chunk_text_is_rejected() -> None:
    provider = FakeEmbeddingProvider()

    service = EmbeddingService(provider)

    try:
        service.embed_chunk(
            create_chunk("")
        )
    except ValueError as error:
        assert "empty text" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError for empty text."
    )