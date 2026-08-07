from pathlib import Path

from infrastructure.parsers.text_parser import TextParser


def main() -> None:

    parser = TextParser()

    document = parser.parse(Path("sample.txt"))

    print()

    print("========== DOCUMENT ==========")

    print(f"Filename : {document.filename}")
    print(f"Type     : {document.document_type.value}")
    print(f"Pages    : {document.page_count}")

    print()

    print(document.pages[0].text)


if __name__ == "__main__":
    main()