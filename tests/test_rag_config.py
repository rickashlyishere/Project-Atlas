from __future__ import annotations

from config.rag import (
    RAG_MAX_CONTEXT_CHARACTERS,
    RAG_MAX_SOURCES,
    RAG_MINIMUM_SCORE,
)


def test_rag_max_sources_is_positive() -> None:
    assert RAG_MAX_SOURCES > 0


def test_rag_max_context_characters_is_positive() -> None:
    assert RAG_MAX_CONTEXT_CHARACTERS > 0


def test_rag_minimum_score_is_valid() -> None:
    assert 0.0 <= RAG_MINIMUM_SCORE <= 1.0


def test_rag_configuration_values() -> None:
    assert RAG_MAX_SOURCES == 5
    assert RAG_MAX_CONTEXT_CHARACTERS == 12000
    assert RAG_MINIMUM_SCORE == 0.35
