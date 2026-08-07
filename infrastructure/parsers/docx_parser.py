from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from docx import Document as DocxDocument

from domain.document import (
    Document,
    DocumentMetadata,
    DocumentType,
    Page,
)

from infrastructure.parsers.base_parser import BaseParser


class DOCXParser(BaseParser):
    """
    Parser for Microsoft Word documents.
    """

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".docx",)

    def _extract(self, file_path: Path) -> Document:

        doc = DocxDocument(file_path)

        paragraphs = [
            p.text
            for p in doc.paragraphs
            if p.text.strip()
        ]

        text = "\n".join(paragraphs)

        return Document(
            id=str(uuid4()),
            filename=file_path.name,
            filepath=file_path,
            document_type=DocumentType.DOCX,
            pages=[
                Page(
                    number=1,
                    text=text,
                )
            ],
            metadata=DocumentMetadata(
                title=doc.core_properties.title or "",
                author=doc.core_properties.author or "",
                subject=doc.core_properties.subject or "",
                keywords=[],
                language="",
            ),
            extracted_text=text,
            page_count=1,
            file_size=file_path.stat().st_size,
        )