from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from domain.document.enums import ChunkType, DocumentType


@dataclass(slots=True)
class DocumentMetadata:
    title: str = ""
    author: str = ""
    subject: str = ""
    keywords: list[str] = field(default_factory=list)
    language: str = ""
    created_at: datetime | None = None
    modified_at: datetime | None = None


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    page_number: int
    chunk_type: ChunkType
    embedding_id: str | None = None


@dataclass(slots=True)
class Page:
    number: int
    text: str
    chunks: list[Chunk] = field(default_factory=list)


@dataclass(slots=True)
class Document:
    id: str
    filename: str
    filepath: Path
    document_type: DocumentType

    pages: list[Page] = field(default_factory=list)

    metadata: DocumentMetadata = field(
        default_factory=DocumentMetadata
    )

    file_size: int = 0

    extracted_text: str = ""

    page_count: int = 0

    created_at: datetime = field(
        default_factory=datetime.now
    )