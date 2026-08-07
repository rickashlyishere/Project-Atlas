from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.exceptions import ParserError
from domain.document import Document


class BaseParser(ABC):
    """
    Base class for all document parsers.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        """Supported file extensions."""
        ...

    def parse(self, file_path: Path) -> Document:
        """
        Validate the file and delegate parsing
        to the concrete parser.
        """

        if not file_path.exists():
            raise ParserError(f"File not found: {file_path}")

        try:
            return self._extract(file_path)

        except ParserError:
            raise

        except Exception as error:
            raise ParserError(
                f"Failed to parse '{file_path.name}'."
            ) from error

    @abstractmethod
    def _extract(self, file_path: Path) -> Document:
        """
        Concrete parser implementation.
        """
        ...