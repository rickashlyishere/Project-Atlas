from __future__ import annotations

from pathlib import Path

from domain.document import Document

from infrastructure.parsers import (
    DOCXParser,
    PDFParser,
    TextParser,
)

from infrastructure.registry.parser_registry import ParserRegistry

from services.storage_service import StorageService


class DocumentService:
    """
    Handles importing and parsing documents.
    """

    def __init__(self) -> None:

        self.registry = ParserRegistry()

        self.storage = StorageService()

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

        return document