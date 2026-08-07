from __future__ import annotations

from pathlib import Path

from domain.document import Document

from infrastructure.parsers import (
    DOCXParser,
    PDFParser,
    TextParser,
)
from infrastructure.registry.parser_registry import ParserRegistry


class DocumentService:
    """
    High-level service responsible for loading documents.
    """

    def __init__(self) -> None:

        self.registry = ParserRegistry()

        self._register_default_parsers()

    def _register_default_parsers(self) -> None:
        """
        Register all built-in parsers.
        """

        self.registry.register(PDFParser())
        self.registry.register(DOCXParser())
        self.registry.register(TextParser())

    def load(self, file_path: Path) -> Document:
        """
        Load a document using the correct parser.
        """

        parser = self.registry.get_parser(file_path)

        return parser.parse(file_path)