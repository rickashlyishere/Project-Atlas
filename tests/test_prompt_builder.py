from services.context_assembler import (
    AssembledContext,
    ContextSource,
)
from services.prompt_builder import (
    GroundedPromptBuilder,
)


def make_context() -> AssembledContext:
    source = ContextSource(
        source_number=1,
        document_id="document-1",
        filename="history.pdf",
        page_number=4,
        chunk_id="chunk-1",
        score=0.91,
        text=(
            "The document states that the event "
            "occurred in 1947."
        ),
    )

    return AssembledContext(
        text=(
            "[Source 1]\n"
            "Document: history.pdf\n"
            "Page: 4\n"
            "Similarity: 0.9100\n\n"
            "The document states that the event "
            "occurred in 1947."
        ),
        sources=[source],
    )


def test_prompt_contains_question() -> None:
    builder = GroundedPromptBuilder()

    prompt = builder.build(
        question="When did the event occur?",
        context=make_context(),
    )

    assert (
        "When did the event occur?"
        in prompt
    )


def test_prompt_contains_document_context() -> None:
    builder = GroundedPromptBuilder()

    prompt = builder.build(
        question="When did the event occur?",
        context=make_context(),
    )

    assert (
        "The document states that the event "
        "occurred in 1947."
        in prompt
    )


def test_prompt_contains_source_reference() -> None:
    builder = GroundedPromptBuilder()

    prompt = builder.build(
        question="When did the event occur?",
        context=make_context(),
    )

    assert "[Source 1]" in prompt


def test_prompt_contains_grounding_instruction() -> None:
    builder = GroundedPromptBuilder()

    prompt = builder.build(
        question="When did the event occur?",
        context=make_context(),
    )

    assert (
        "Do not invent facts"
        in prompt
    )


def test_empty_question_is_rejected() -> None:
    builder = GroundedPromptBuilder()

    try:
        builder.build(
            question="",
            context=make_context(),
        )
    except ValueError as error:
        assert "question" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError."
    )


def test_whitespace_question_is_rejected() -> None:
    builder = GroundedPromptBuilder()

    try:
        builder.build(
            question="   ",
            context=make_context(),
        )
    except ValueError as error:
        assert "question" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError."
    )


def test_empty_context_is_rejected() -> None:
    builder = GroundedPromptBuilder()

    context = AssembledContext(
        text="",
        sources=[],
    )

    try:
        builder.build(
            question="What happened?",
            context=context,
        )
    except ValueError as error:
        assert "context" in str(error).lower()
        return

    raise AssertionError(
        "Expected ValueError."
    )