from pathlib import Path

from infrastructure.parsers.docx_parser import DOCXParser


def main() -> None:
    parser = DOCXParser()

    document = parser.parse(Path("sample.docx"))

    print("\n========== DOCUMENT ==========")
    print(f"Filename : {document.filename}")
    print(f"Type     : {document.document_type.value}")
    print(f"Pages    : {document.page_count}")
    print(f"Author   : {document.metadata.author}")
    print(f"Title    : {document.metadata.title}")

    print("\n========== CONTENT ==========\n")

    print(document.extracted_text)


if __name__ == "__main__":
    main()