from pathlib import Path

from services.document_service import DocumentService


def print_document(document) -> None:

    print("\n" + "=" * 60)
    print(document.filename)
    print("=" * 60)

    print(f"Type      : {document.document_type.value}")
    print(f"Pages     : {document.page_count}")
    print(f"Size      : {document.file_size} bytes")
    print(f"Title     : {document.metadata.title}")
    print(f"Author    : {document.metadata.author}")

    print("\n========== CONTENT ==========\n")

    print(document.extracted_text[:1000])


def main() -> None:

    service = DocumentService()

    document = service.load(
        Path("sample.pdf")
    )

    print_document(document)


if __name__ == "__main__":
    main()