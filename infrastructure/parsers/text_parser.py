from __future__ import annotations

from pathlib import Path
from uuid import uuid4

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

    def _extract(self, file_path: Path) -> Document:

        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        return Document(
            id=str(uuid4()),
            filename=file_path.name,
            filepath=file_path,
            document_type=DocumentType.TXT,
            pages=[
                Page(
                    number=1,
                    text=text,
                )
            ],
            metadata=DocumentMetadata(),
            extracted_text=text,
            page_count=1,
            file_size=file_path.stat().st_size,
        )