from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

import pymupdf

from domain.document import (
    Document,
    DocumentMetadata,
    DocumentType,
    Page,
)

from infrastructure.parsers.base_parser import BaseParser


class PDFParser(BaseParser):
    """
    Parser for PDF documents.

    Text-based PDF pages are extracted directly with PyMuPDF.

    If a page contains no usable text, the page is rendered
    as an image and processed with Tesseract OCR.

    This supports both normal PDFs and scanned PDFs.
    """

    TESSERACT_CANDIDATES = (
        "tesseract",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    )

    @property
    def supported_extensions(
        self,
    ) -> tuple[str, ...]:
        return (".pdf",)

    def _extract(
        self,
        file_path: Path,
    ) -> Document:

        with pymupdf.open(file_path) as pdf:

            metadata = pdf.metadata or {}

            pages: list[Page] = []
            full_text: list[str] = []

            page_count = len(pdf)

            for page_index in range(page_count):

                page = pdf.load_page(
                    page_index
                )

                page_number = page_index + 1

                text = page.get_text()

                if isinstance(text, str):
                    text = self._normalize_text(
                        text
                    )
                else:
                    text = ""

                # ------------------------------------------------
                # OCR FALLBACK
                # ------------------------------------------------

                if not text:

                    text = self._ocr_page(
                        page
                    )

                pages.append(
                    Page(
                        number=page_number,
                        text=text,
                    )
                )

                if text:

                    full_text.append(
                        text
                    )

            return Document(
                id=str(uuid4()),
                filename=file_path.name,
                filepath=file_path,
                document_type=DocumentType.PDF,
                pages=pages,
                metadata=DocumentMetadata(
                    title=self._metadata_string(
                        metadata,
                        "title",
                    ),
                    author=self._metadata_string(
                        metadata,
                        "author",
                    ),
                    subject=self._metadata_string(
                        metadata,
                        "subject",
                    ),
                    keywords=[],
                    language="",
                ),
                extracted_text="\n\n".join(
                    full_text
                ),
                page_count=page_count,
                file_size=file_path.stat().st_size,
            )

    @staticmethod
    def _metadata_string(
        metadata: dict,
        key: str,
    ) -> str:
        """
        Safely extract a string metadata value.

        PyMuPDF's metadata typing permits several value types,
        while Atlas's DocumentMetadata expects strings.
        """

        value = metadata.get(key)

        if isinstance(value, str):

            return value

        if value is None:

            return ""

        return str(value)

    @classmethod
    def _find_tesseract(
        cls,
    ) -> str:
        """
        Find the Tesseract executable.
        """

        for candidate in cls.TESSERACT_CANDIDATES:

            if candidate == "tesseract":

                executable = shutil.which(
                    candidate
                )

                if executable:

                    return executable

                continue

            path = Path(
                candidate
            )

            if path.is_file():

                return str(path)

        raise RuntimeError(
            "Tesseract OCR is required to process "
            "scanned PDF pages, but tesseract.exe "
            "could not be found. Install Tesseract OCR "
            "or add it to PATH."
        )

    @classmethod
    def _ocr_page(
        cls,
        page: pymupdf.Page,
    ) -> str:
        """
        Render a PDF page and extract text with Tesseract.
        """

        tesseract = cls._find_tesseract()

        # 300 DPI rendering.
        zoom = 300 / 72

        matrix = pymupdf.Matrix(
            zoom,
            zoom,
        )

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        temp_path: Path | None = None

        try:

            with tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False,
            ) as temporary_file:

                temp_path = Path(
                    temporary_file.name
                )

            pixmap.save(
                str(temp_path)
            )

            result = subprocess.run(
                [
                    tesseract,
                    str(temp_path),
                    "stdout",
                    "--psm",
                    "3",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

            if result.returncode != 0:

                error_message = (
                    result.stderr.strip()
                    or "Unknown Tesseract error."
                )

                raise RuntimeError(
                    "Tesseract OCR failed: "
                    f"{error_message}"
                )

            return cls._normalize_text(
                result.stdout
            )

        finally:

            if (
                temp_path is not None
                and temp_path.exists()
            ):

                temp_path.unlink(
                    missing_ok=True
                )

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Normalize extracted/OCR text while preserving
        meaningful blank lines.
        """

        lines: list[str] = []

        previous_blank = False

        for line in text.splitlines():

            stripped = line.strip()

            if not stripped:

                if not previous_blank:

                    lines.append("")

                previous_blank = True

                continue

            lines.append(
                stripped
            )

            previous_blank = False

        return "\n".join(
            lines
        ).strip()