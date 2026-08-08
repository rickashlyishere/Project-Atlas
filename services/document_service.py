from __future__ import annotations

from pathlib import Path

from domain.document import ChunkType, Document

from infrastructure.database import (
    ChunkRepository,
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
from services.chunk_service import ChunkService
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

        self.repository = DocumentRepository(
            self.database
        )

        self.chunk_repository = ChunkRepository(
            self.database
        )

        self.chunk_service = ChunkService(
            strategy=ChunkType.FIXED,
            chunk_size=1000,
            chunk_overlap=200,
        )

        self._register_default_parsers()

    def _register_default_parsers(self) -> None:

        self.registry.register(PDFParser())
        self.registry.register(DOCXParser())
        self.registry.register(TextParser())

    def load(
        self,
        file_path: Path,
    ) -> Document:
        """
        Parse, store, chunk, and persist a document.
        """

        parser = self.registry.get_parser(file_path)

        document = parser.parse(file_path)

        self.storage.save(
            file_path,
            document,
        )

        self.repository.add(document)

        chunks = self.chunk_service.chunk_document(
            document
        )

        self.chunk_repository.add_many(
            document_id=document.id,
            chunks=chunks,
        )

        return document

    def list_documents(self):

        return self.repository.list_all()

    def get_chunks(
        self,
        document_id: str,
    ):
        """
        Return all persisted chunks for a document.
        """

        return self.chunk_repository.get_by_document(
            document_id
        )