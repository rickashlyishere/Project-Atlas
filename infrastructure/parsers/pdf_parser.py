from __future__ import annotations

from pathlib import Path

import pymupdf

from core.exceptions import ParserError
from domain.document import (
    Document,
    DocumentMetadata,
    DocumentType,
    Page,
)
from domain.factories import DocumentFactory
from infrastructure.parsers.base_parser import BaseParser


class PDFParser(BaseParser):
    """
    Parser for PDF documents using PyMuPDF.
    """

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".pdf",)

    def parse(self, file_path: Path) -> Document:

        if not file_path.exists():
            raise ParserError(f"File not found: {file_path}")

        try:

            with pymupdf.open(file_path) as pdf:

                metadata = pdf.metadata or {}

                pages: list[Page] = []

                for page_number, page in enumerate(pdf, start=1):

                    pages.append(
                        Page(
                            number=page_number,
                            text=page.get_text(),
                        )
                    )

                return DocumentFactory.create(
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
                )

        except Exception as error:

            raise ParserError(
                f"Failed to parse PDF '{file_path.name}'."
            ) from error