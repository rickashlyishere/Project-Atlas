from __future__ import annotations

import re
from uuid import uuid4

from domain.document import Chunk, ChunkType, Document


class ChunkService:
    """
    Split documents into chunks using a configurable strategy.

    Supported strategies:
        - FIXED
        - PARAGRAPH
        - SENTENCE

    SEMANTIC chunking is intentionally reserved for a later sprint.
    """

    def __init__(
        self,
        strategy: ChunkType = ChunkType.FIXED,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        if strategy == ChunkType.SEMANTIC:
            raise ValueError(
                "Semantic chunking is not implemented yet."
            )

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        document: Document,
    ) -> list[Chunk]:
        """
        Chunk every page in a document.

        Pages are processed independently so page boundaries
        remain available for retrieval and citations.
        """

        chunks: list[Chunk] = []

        for page in document.pages:
            page_chunks = self.chunk_page(
                page_number=page.number,
                text=page.text,
            )

            page.chunks = page_chunks
            chunks.extend(page_chunks)

        return chunks

    def chunk_page(
        self,
        page_number: int,
        text: str,
    ) -> list[Chunk]:
        """
        Chunk one page according to the configured strategy.
        """

        text = self._normalize_text(text)

        if not text:
            return []

        if self.strategy == ChunkType.FIXED:
            return self._chunk_fixed(
                page_number=page_number,
                text=text,
            )

        if self.strategy == ChunkType.PARAGRAPH:
            return self._chunk_paragraphs(
                page_number=page_number,
                text=text,
            )

        if self.strategy == ChunkType.SENTENCE:
            return self._chunk_sentences(
                page_number=page_number,
                text=text,
            )

        raise ValueError(
            f"Unsupported chunking strategy: {self.strategy}"
        )

    def _chunk_fixed(
        self,
        page_number: int,
        text: str,
    ) -> list[Chunk]:
        """
        Split text into fixed-size overlapping chunks.
        """

        chunks: list[Chunk] = []

        step = self.chunk_size - self.chunk_overlap
        start = 0

        while start < len(text):
            end = min(
                start + self.chunk_size,
                len(text),
            )

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    self._create_chunk(
                        text=chunk_text,
                        page_number=page_number,
                        chunk_type=ChunkType.FIXED,
                    )
                )

            if end >= len(text):
                break

            start += step

        return chunks

    def _chunk_paragraphs(
        self,
        page_number: int,
        text: str,
    ) -> list[Chunk]:
        """
        Group paragraphs into chunks without unnecessarily
        breaking paragraph boundaries.

        When a paragraph is larger than chunk_size, it is
        split using the fixed-size algorithm.
        """

        paragraphs = self._split_paragraphs(text)

        if not paragraphs:
            return []

        return self._group_text_units(
            page_number=page_number,
            units=paragraphs,
            chunk_type=ChunkType.PARAGRAPH,
        )

    def _chunk_sentences(
        self,
        page_number: int,
        text: str,
    ) -> list[Chunk]:
        """
        Group sentences into chunks while preserving
        sentence boundaries.
        """

        sentences = self._split_sentences(text)

        if not sentences:
            return []

        return self._group_text_units(
            page_number=page_number,
            units=sentences,
            chunk_type=ChunkType.SENTENCE,
        )

    def _group_text_units(
        self,
        page_number: int,
        units: list[str],
        chunk_type: ChunkType,
    ) -> list[Chunk]:
        """
        Group paragraphs or sentences until chunk_size is reached.

        Overlap is created by carrying the final units of the
        previous chunk into the next chunk.
        """

        chunks: list[Chunk] = []
        index = 0

        while index < len(units):
            current_units: list[str] = []
            current_length = 0

            start_index = index

            while index < len(units):
                unit = units[index]
                additional_length = len(unit)

                if current_units:
                    additional_length += 2

                if (
                    current_units
                    and current_length + additional_length
                    > self.chunk_size
                ):
                    break

                current_units.append(unit)
                current_length += additional_length
                index += 1

                if current_length >= self.chunk_size:
                    break

            if not current_units:
                oversized = units[index]

                fixed_chunks = self._split_oversized_unit(
                    page_number=page_number,
                    text=oversized,
                    chunk_type=chunk_type,
                )

                chunks.extend(fixed_chunks)
                index += 1
                continue

            chunk_text = "\n\n".join(current_units)

            chunks.append(
                self._create_chunk(
                    text=chunk_text,
                    page_number=page_number,
                    chunk_type=chunk_type,
                )
            )

            if index >= len(units):
                break

            overlap_length = 0
            overlap_start = index

            while overlap_start > start_index:
                candidate = units[overlap_start - 1]

                candidate_length = len(candidate)

                if overlap_length:
                    candidate_length += 2

                if (
                    overlap_length + candidate_length
                    > self.chunk_overlap
                ):
                    break

                overlap_length += candidate_length
                overlap_start -= 1

            if overlap_start == index:
                continue

            index = overlap_start

        return chunks

    def _split_oversized_unit(
        self,
        page_number: int,
        text: str,
        chunk_type: ChunkType,
    ) -> list[Chunk]:
        """
        Split an oversized paragraph or sentence into smaller chunks.

        The resulting chunks retain the original strategy type.
        """

        chunks: list[Chunk] = []

        step = self.chunk_size - self.chunk_overlap
        start = 0

        while start < len(text):
            end = min(
                start + self.chunk_size,
                len(text),
            )

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    self._create_chunk(
                        text=chunk_text,
                        page_number=page_number,
                        chunk_type=chunk_type,
                    )
                )

            if end >= len(text):
                break

            start += step

        return chunks

    @staticmethod
    def _split_paragraphs(
        text: str,
    ) -> list[str]:
        """
        Split text using blank lines as paragraph boundaries.
        """

        paragraphs = re.split(
            r"\n\s*\n+",
            text,
        )

        return [
            paragraph.strip()
            for paragraph in paragraphs
            if paragraph.strip()
        ]

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[str]:
        """
        Split text at common sentence boundaries.

        This intentionally uses a lightweight regex instead of
        adding an NLP dependency.
        """

        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9\"'])",
            text,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Normalize whitespace while preserving blank lines.

        Blank lines are intentionally retained because paragraph
        chunking uses them as paragraph boundaries.
        """

        normalized_lines: list[str] = []

        previous_blank = False

        for line in text.splitlines():
            stripped = line.strip()

            if not stripped:
                if not previous_blank:
                    normalized_lines.append("")

                previous_blank = True
                continue

            normalized_lines.append(stripped)
            previous_blank = False

        return "\n".join(normalized_lines).strip()

    @staticmethod
    def _create_chunk(
        text: str,
        page_number: int,
        chunk_type: ChunkType,
    ) -> Chunk:
        """
        Create a domain Chunk object.
        """

        return Chunk(
            id=str(uuid4()),
            text=text,
            page_number=page_number,
            chunk_type=chunk_type,
        )
    