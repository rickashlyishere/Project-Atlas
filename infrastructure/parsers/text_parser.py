from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from core.exceptions import ParserError
from domain.document import (
    Document,
    DocumentMetadata,
    DocumentType,
    Page,
)

from infrastructure.parsers.base_parser import BaseParser


class TextParser(BaseParser):
    """
    Parser for plain text files.
    """

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".txt",)

    def parse(self, file_path: Path) -> Document:

        if not file_path.exists():
            raise ParserError(f"File not found: {file_path}")

        try:

            text = file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            page = Page(
                number=1,
                text=text,
            )

            return Document(
                id=str(uuid4()),
                filename=file_path.name,
                filepath=file_path,
                document_type=DocumentType.TXT,
                pages=[page],
                metadata=DocumentMetadata(),
                extracted_text=text,
                page_count=1,
                file_size=file_path.stat().st_size,
            )

        except Exception as error:

            raise ParserError(
                f"Failed to parse '{file_path.name}'."
            ) from error