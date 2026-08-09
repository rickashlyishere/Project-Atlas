from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextSource:
    """
    A source used to build LLM context.
    """

    source_number: int
    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    score: float
    text: str


@dataclass(frozen=True)
class AssembledContext:
    """
    LLM-ready context plus structured source metadata.
    """

    text: str
    sources: list[ContextSource]


class ContextAssembler:
    """
    Converts ranked semantic-search results into bounded,
    ordered context for an LLM.

    This class deliberately knows nothing about:
    - Streamlit
    - Ollama
    - OpenAI
    - Gemini
    - any other LLM provider
    """

    def __init__(
        self,
        max_sources: int = 5,
        max_characters: int = 12000,
        minimum_score: float = 0.0,
    ) -> None:

        if max_sources <= 0:
            raise ValueError(
                "max_sources must be greater than zero."
            )

        if max_characters <= 0:
            raise ValueError(
                "max_characters must be greater than zero."
            )

        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError(
                "minimum_score must be between 0.0 and 1.0."
            )

        self.max_sources = max_sources
        self.max_characters = max_characters
        self.minimum_score = minimum_score

    def assemble(
        self,
        results: list[dict[str, Any]],
    ) -> AssembledContext:
        """
        Build bounded context from ranked search results.
        """

        sources: list[ContextSource] = []
        context_parts: list[str] = []

        current_characters = 0

        for result in results:

            if len(sources) >= self.max_sources:
                break

            text = str(
                result.get("text", "")
            ).strip()

            if not text:
                continue

            score = float(
                result.get("score", 0.0)
            )

            if score < self.minimum_score:
                continue

            filename = str(
                result.get(
                    "filename",
                    "Unknown document",
                )
            )

            document_id = str(
                result.get(
                    "document_id",
                    "",
                )
            )

            chunk_id = str(
                result.get(
                    "chunk_id",
                    "",
                )
            )

            page_number = int(
                result.get(
                    "page_number",
                    0,
                )
            )

            source_number = (
                len(sources) + 1
            )

            header = (
                f"[Source {source_number}]\n"
                f"Document: {filename}\n"
                f"Page: {page_number}\n"
                f"Similarity: {score:.4f}\n\n"
            )

            separator = "\n\n"

            available_characters = (
                self.max_characters
                - current_characters
            )

            if available_characters <= 0:
                break

            full_source = (
                header
                + text
            )

            if context_parts:
                full_source = (
                    separator
                    + full_source
                )

            if len(full_source) > available_characters:

                if not sources:
                    remaining_text = (
                        available_characters
                        - len(header)
                    )

                    if remaining_text <= 0:
                        break

                    truncated_text = (
                        text[:remaining_text]
                        .rstrip()
                    )

                    full_source = (
                        header
                        + truncated_text
                    )

                else:
                    break

            context_parts.append(
                full_source
            )

            current_characters += len(
                full_source
            )

            sources.append(
                ContextSource(
                    source_number=source_number,
                    document_id=document_id,
                    filename=filename,
                    page_number=page_number,
                    chunk_id=chunk_id,
                    score=score,
                    text=text,
                )
            )

        return AssembledContext(
            text="".join(context_parts),
            sources=sources,
        )