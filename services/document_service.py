from __future__ import annotations

from pathlib import Path

from domain.document import Document

from infrastructure.database import (
    Database,
    DocumentRepository,
    SchemaManager,
)
from infrastructure.parsers import (
    DOCXParser,
    PDFParser,
    TextParser,
)
from infrastructure.registry.parser_registry import ParserRegistry
from services.storage_service import StorageService


class DocumentService:
    """
    High-level service responsible for importing documents.
    """

    def __init__(self) -> None:

        self.registry = ParserRegistry()

        self.storage = StorageService()

        self.database = Database()

        SchemaManager(self.database).initialize()

        self.repository = DocumentRepository(self.database)

        self._register_default_parsers()

    def _register_default_parsers(self) -> None:

        self.registry.register(PDFParser())
        self.registry.register(DOCXParser())
        self.registry.register(TextParser())

    def load(
        self,
        file_path: Path,
    ) -> Document:

        parser = self.registry.get_parser(file_path)

        document = parser.parse(file_path)

        self.storage.save(
            file_path,
            document,
        )

        self.repository.add(document)

        return document

    def list_documents(self):

        return self.repository.list_all()