from __future__ import annotations

from core.exceptions.base import AtlasError


class ParserError(AtlasError):
    """Raised when document parsing fails."""

    pass


class UnsupportedFileTypeError(ParserError):
    """Raised for unsupported file types."""

    pass


class CorruptedDocumentError(ParserError):
    """Raised when a document cannot be parsed."""

    pass