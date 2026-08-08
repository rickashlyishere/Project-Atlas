import pytest

from services.vector_search_service import (
    VectorSearchService,
)


def test_identical_vectors_have_similarity_one() -> None:
    service = VectorSearchService()

    score = service.cosine_similarity(
        [1.0, 0.0],
        [1.0, 0.0],
    )

    assert score == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero() -> None:
    service = VectorSearchService()

    score = service.cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert score == pytest.approx(0.0)


def test_search_returns_highest_similarity_first() -> None:
    service = VectorSearchService()

    candidates = [
        {
            "id": "low",
            "vector": [0.0, 1.0],
        },
        {
            "id": "high",
            "vector": [1.0, 0.0],
        },
        {
            "id": "medium",
            "vector": [0.7, 0.7],
        },
    ]

    results = service.search(
        query_vector=[1.0, 0.0],
        candidates=candidates,
        top_k=3,
    )

    assert results[0]["id"] == "high"
    assert results[1]["id"] == "medium"
    assert results[2]["id"] == "low"


def test_top_k_limits_results() -> None:
    service = VectorSearchService()

    candidates = [
        {
            "id": "one",
            "vector": [1.0, 0.0],
        },
        {
            "id": "two",
            "vector": [0.9, 0.1],
        },
        {
            "id": "three",
            "vector": [0.8, 0.2],
        },
    ]

    results = service.search(
        query_vector=[1.0, 0.0],
        candidates=candidates,
        top_k=2,
    )

    assert len(results) == 2


def test_dimension_mismatch_is_rejected() -> None:
    service = VectorSearchService()

    with pytest.raises(ValueError):
        service.search(
            query_vector=[1.0, 0.0],
            candidates=[
                {
                    "id": "bad",
                    "vector": [1.0, 0.0, 0.0],
                }
            ],
        )