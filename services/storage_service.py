from __future__ import annotations

import json
import shutil
from pathlib import Path

from domain.document import Document


class StorageService:
    """
    Handles persistent storage of Atlas documents.
    """

    def __init__(self) -> None:

        self.storage_root = Path("data/documents")

        self.storage_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        source_file: Path,
        document: Document,
    ) -> Path:
        """
        Save the original document and its metadata.
        """

        document_directory = self.storage_root / document.id

        document_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = document_directory / source_file.name

        shutil.copy2(
            source_file,
            destination,
        )

        metadata = {
            "id": document.id,
            "filename": document.filename,
            "filepath": str(destination),
            "document_type": document.document_type.value,
            "page_count": document.page_count,
            "file_size": document.file_size,
            "title": document.metadata.title,
            "author": document.metadata.author,
            "subject": document.metadata.subject,
        }

        with open(
            document_directory / "metadata.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
            )

        return destination