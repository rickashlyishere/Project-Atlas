from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from domain.document import Document


class BaseParser(ABC):
    """
    Base class for every document parser.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        """Return supported extensions."""
        ...

    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        """
        Parse a document and return
        a Document model.
        """
        ...