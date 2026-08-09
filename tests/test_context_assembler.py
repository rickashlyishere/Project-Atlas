from services.context_assembler import (
    ContextAssembler,
)


def make_result(
    *,
    text: str,
    score: float,
    filename: str = "test.pdf",
    page_number: int = 1,
    document_id: str = "document-1",
    chunk_id: str = "chunk-1",
) -> dict:

    return {
        "text": text,
        "score": score,
        "filename": filename,
        "page_number": page_number,
        "document_id": document_id,
        "chunk_id": chunk_id,
    }


def test_context_assembler_creates_sources() -> None:

    assembler = ContextAssembler()

    results = [
        make_result(
            text="Python is a programming language.",
            score=0.95,
        ),
        make_result(
            text="Linux is an operating system.",
            score=0.80,
            page_number=2,
        ),
    ]

    context = assembler.assemble(
        results
    )

    assert len(
        context.sources
    ) == 2

    assert (
        context.sources[0].source_number
        == 1
    )

    assert (
        context.sources[0].filename
        == "test.pdf"
    )

    assert (
        context.sources[0].page_number
        == 1
    )

    assert (
        context.sources[1].source_number
        == 2
    )


def test_context_contains_source_metadata() -> None:

    assembler = ContextAssembler()

    results = [
        make_result(
            text="Python is useful.",
            score=0.91,
            filename="python.pdf",
            page_number=7,
        )
    ]

    context = assembler.assemble(
        results
    )

    assert (
        "[Source 1]"
        in context.text
    )

    assert (
        "Document: python.pdf"
        in context.text
    )

    assert (
        "Page: 7"
        in context.text
    )

    assert (
        "Similarity: 0.9100"
        in context.text
    )

    assert (
        "Python is useful."
        in context.text
    )


def test_empty_results_create_empty_context() -> None:

    assembler = ContextAssembler()

    context = assembler.assemble(
        []
    )

    assert context.text == ""
    assert context.sources == []


def test_empty_text_results_are_skipped() -> None:

    assembler = ContextAssembler()

    results = [
        make_result(
            text="",
            score=0.99,
        ),
        make_result(
            text="Useful content.",
            score=0.80,
        ),
    ]

    context = assembler.assemble(
        results
    )

    assert len(
        context.sources
    ) == 1

    assert (
        context.sources[0].text
        == "Useful content."
    )


def test_minimum_score_filters_results() -> None:

    assembler = ContextAssembler(
        minimum_score=0.80,
    )

    results = [
        make_result(
            text="Weak result.",
            score=0.70,
        ),
        make_result(
            text="Strong result.",
            score=0.90,
        ),
    ]

    context = assembler.assemble(
        results
    )

    assert len(
        context.sources
    ) == 1

    assert (
        context.sources[0].text
        == "Strong result."
    )


def test_max_sources_limits_context() -> None:

    assembler = ContextAssembler(
        max_sources=2,
    )

    results = [
        make_result(
            text="First.",
            score=0.90,
        ),
        make_result(
            text="Second.",
            score=0.80,
        ),
        make_result(
            text="Third.",
            score=0.70,
        ),
    ]

    context = assembler.assemble(
        results
    )

    assert len(
        context.sources
    ) == 2

    assert (
        "First."
        in context.text
    )

    assert (
        "Second."
        in context.text
    )

    assert (
        "Third."
        not in context.text
    )


def test_max_characters_bounds_context() -> None:

    assembler = ContextAssembler(
        max_sources=5,
        max_characters=100,
    )

    results = [
        make_result(
            text=(
                "This is a deliberately long "
                "piece of text that should be "
                "bounded by the context limit."
            ),
            score=0.95,
        )
    ]

    context = assembler.assemble(
        results
    )

    assert len(
        context.text
    ) <= 100


def test_invalid_max_sources_is_rejected() -> None:

    try:
        ContextAssembler(
            max_sources=0,
        )
    except ValueError as error:
        assert "max_sources" in str(error)
        return

    raise AssertionError(
        "Expected ValueError."
    )


def test_invalid_max_characters_is_rejected() -> None:

    try:
        ContextAssembler(
            max_characters=0,
        )
    except ValueError as error:
        assert "max_characters" in str(error)
        return

    raise AssertionError(
        "Expected ValueError."
    )


def test_invalid_minimum_score_is_rejected() -> None:

    try:
        ContextAssembler(
            minimum_score=1.5,
        )
    except ValueError as error:
        assert "minimum_score" in str(error)
        return

    raise AssertionError(
        "Expected ValueError."
    )