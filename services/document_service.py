from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from domain.document import ChunkType, Document

from infrastructure.database import (
    ChunkRepository,
    Database,
    DocumentRepository,
    EmbeddingRepository,
    SchemaManager,
)
from infrastructure.embeddings import (
    SentenceTransformerProvider,
)
from infrastructure.parsers import (
    DOCXParser,
    ImageParser,
    PDFParser,
    PPTXParser,
    TextParser,
)
from infrastructure.registry.parser_registry import (
    ParserRegistry,
)

from services.chunk_service import ChunkService
from services.embedding_service import EmbeddingService
from services.storage_service import StorageService


class DocumentService:
    """
    High-level service responsible for importing documents.

    Dependencies can be injected for testing or for swapping
    implementations.
    """

    def __init__(
        self,
        database: Database | None = None,
        chunk_service: ChunkService | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:

        self.registry = ParserRegistry()

        self.storage = StorageService()

        self.database = database or Database()

        SchemaManager(
            self.database
        ).initialize()

        self.repository = DocumentRepository(
            self.database
        )

        # Backfill hashes for documents created by older
        # versions of Atlas.
        self.repository.backfill_content_hashes()

        self.chunk_repository = ChunkRepository(
            self.database
        )

        self.embedding_repository = EmbeddingRepository(
            self.database
        )

        self.chunk_service = (
            chunk_service
            if chunk_service is not None
            else ChunkService(
                strategy=ChunkType.FIXED,
                chunk_size=1000,
                chunk_overlap=200,
            )
        )

        if embedding_service is None:
            embedding_service = EmbeddingService(
                provider=SentenceTransformerProvider(),
                repository=self.embedding_repository,
            )

        self.embedding_service = embedding_service

        self._register_default_parsers()

    def _register_default_parsers(self) -> None:
        """
        Register all supported document parsers.
        """

        self.registry.register(PDFParser())
        self.registry.register(DOCXParser())
        self.registry.register(PPTXParser())
        self.registry.register(TextParser())
        self.registry.register(ImageParser())

    @staticmethod
    def _calculate_file_hash(
        file_path: Path,
    ) -> str:
        """
        Calculate SHA-256 for the source file.
        """

        digest = hashlib.sha256()

        with file_path.open(
            "rb"
        ) as file:
            for block in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(block)

        return digest.hexdigest()

    def load(
        self,
        file_path: Path,
    ) -> Document:
        """
        Parse, store, chunk, and embed a document.

        If the exact same file content has already been
        indexed, return the existing document instead of
        creating duplicate chunks and embeddings.
        """

        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Document file does not exist: {file_path}"
            )

        content_hash = (
            self._calculate_file_hash(
                file_path
            )
        )

        existing = (
            self.repository.get_by_content_hash(
                content_hash
            )
        )

        parser = self.registry.get_parser(
            file_path
        )

        # Parse even when the document already exists so the
        # returned object remains a proper domain Document.
        document = parser.parse(
            file_path
        )

        if existing is not None:
            document.id = str(
                existing["id"]
            )

            document.filepath = Path(
                str(existing["filepath"])
            )

            document.created_at = (
                datetime.fromisoformat(
                    str(existing["created_at"])
                )
            )

            document.content_hash = (
                content_hash
            )

            return document

        document.content_hash = content_hash

        self.storage.save(
            file_path,
            document,
        )

        self.repository.add(
            document
        )

        chunks = self.chunk_service.chunk_document(
            document
        )

        self.chunk_repository.add_many(
            document_id=document.id,
            chunks=chunks,
        )

        if chunks:
            self.embedding_service.embed_and_persist(
                chunks
            )

            # Persist the generated embedding IDs
            # back onto the chunk records.
            self.chunk_repository.add_many(
                document_id=document.id,
                chunks=chunks,
            )

        return document

    def list_documents(self):
        """
        Return all stored documents.
        """

        return self.repository.list_all()

    def get_chunks(
        self,
        document_id: str,
    ):
        """
        Return all chunks belonging to a document.
        """

        return self.chunk_repository.get_by_document(
            document_id
        )

    def get_embedding(
        self,
        chunk_id: str,
    ):
        """
        Return the embedding belonging to a chunk.
        """

        return self.embedding_repository.get_by_chunk(
            chunk_id
        )
