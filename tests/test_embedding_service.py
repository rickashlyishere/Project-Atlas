from domain.document import Chunk, ChunkType
from domain.embeddings import EmbeddingProvider
from services.embedding_service import EmbeddingService


class FakeEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic provider used for unit testing.

    It does not load an actual ML model.
    """

    model_name = "fake-model"

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


class FakeEmbeddingRepository:
    """
    In-memory repository used to test persistence behavior
    without requiring SQLite.
    """

    def __init__(self) -> None:
        self.embeddings: list[dict] = []

    def add_many(
        self,
        embeddings: list[dict],
    ) -> None:
        self.embeddings.extend(embeddings)


def create_service() -> EmbeddingService:
    provider = FakeEmbeddingProvider()
    repository = FakeEmbeddingRepository()

    return EmbeddingService(
        provider=provider,
        repository=repository,
    )


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
    service = create_service()

    assert service.dimension == 3


def test_embed_chunk() -> None:
    service = create_service()

    vector = service.embed_chunk(
        create_chunk()
    )

    assert vector == [
        1.0,
        0.0,
        0.0,
    ]


def test_embed_chunks() -> None:
    service = create_service()

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
    service = create_service()

    assert service.embed_chunks([]) == []


def test_empty_chunk_text_is_rejected() -> None:
    service = create_service()

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


def test_embed_and_persist() -> None:
    provider = FakeEmbeddingProvider()
    repository = FakeEmbeddingRepository()

    service = EmbeddingService(
        provider=provider,
        repository=repository,
    )

    chunks = [
        create_chunk("First chunk."),
        create_chunk("Second chunk."),
    ]

    embeddings = service.embed_and_persist(
        chunks
    )

    assert len(embeddings) == 2
    assert len(repository.embeddings) == 2

    assert all(
        embedding["model_name"] == "fake-model"
        for embedding in embeddings
    )

    assert all(
        chunk.embedding_id is not None
        for chunk in chunks
    )