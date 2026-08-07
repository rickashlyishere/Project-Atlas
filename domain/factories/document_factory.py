from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from domain.document import (
    Document,
    DocumentMetadata,
    DocumentType,
    Page,
)


class DocumentFactory:
    """
    Factory responsible for creating Document objects.
    """

    @staticmethod
    def create(
        *,
        filename: str,
        filepath: Path,
        document_type: DocumentType,
        pages: list[Page],
        metadata: DocumentMetadata,
    ) -> Document:

        extracted_text = "\n".join(page.text for page in pages)

        return Document(
            id=str(uuid4()),
            filename=filename,
            filepath=filepath,
            document_type=document_type,
            pages=pages,
            metadata=metadata,
            extracted_text=extracted_text,
            page_count=len(pages),
            file_size=filepath.stat().st_size,
        )