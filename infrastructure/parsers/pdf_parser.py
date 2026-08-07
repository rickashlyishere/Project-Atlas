from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pymupdf

from domain.document import (
    Document,
    DocumentMetadata,
    DocumentType,
    Page,
)

from infrastructure.parsers.base_parser import BaseParser


class PDFParser(BaseParser):
    """
    Parser for PDF documents.
    """

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".pdf",)

    def _extract(self, file_path: Path) -> Document:

        with pymupdf.open(file_path) as pdf:

            metadata = pdf.metadata or {}

            pages: list[Page] = []

            full_text: list[str] = []

            for page_number, page in enumerate(pdf, start=1):

                text = page.get_text()

                pages.append(
                    Page(
                        number=page_number,
                        text=text,
                    )
                )

                full_text.append(text)

            return Document(
                id=str(uuid4()),
                filename=file_path.name,
                filepath=file_path,
                document_type=DocumentType.PDF,
                pages=pages,
                metadata=DocumentMetadata(
                    title=metadata.get("title", ""),
                    author=metadata.get("author", ""),
                    subject=metadata.get("subject", ""),
                    keywords=[],
                    language="",
                ),
                extracted_text="\n".join(full_text),
                page_count=len(pages),
                file_size=file_path.stat().st_size,
            )