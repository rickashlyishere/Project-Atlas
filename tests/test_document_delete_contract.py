from __future__ import annotations

from pathlib import Path

from infrastructure.database import Database
from services.document_service import DocumentService


def create_service() -> DocumentService:
    database = Database(":memory:")

    return DocumentService(
        database=database,
    )


def create_text_file(
    tmp_path: Path,
) -> Path:
    file_path = (
        tmp_path
        / "release_test.txt"
    )

    file_path.write_text(
        "Atlas release hardening test.",
        encoding="utf-8",
    )

    return file_path


def test_delete_contract_removes_document_and_index(
    tmp_path: Path,
) -> None:
    service = create_service()

    file_path = create_text_file(
        tmp_path
    )

    document = service.load(
        file_path
    )

    assert len(
        service.list_documents()
    ) == 1

    chunks = service.get_chunks(
        document.id
    )

    assert len(chunks) > 0

    assert service.delete(
        document.id
    ) is True

    assert (
        service.list_documents()
        == []
    )

    assert (
        service.get_chunks(
            document.id
        )
        == []
    )

    assert file_path.exists()