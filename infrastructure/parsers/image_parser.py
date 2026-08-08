from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytesseract
from PIL import Image, UnidentifiedImageError

from core.exceptions import OCRError
from domain.document import (
    Document,
    DocumentMetadata,
    DocumentType,
    Page,
)
from infrastructure.parsers.base_parser import BaseParser


class ImageParser(BaseParser):
    """
    Parser for image documents using Tesseract OCR.
    """

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (
            ".png",
            ".jpg",
            ".jpeg",
        )

    def _extract(self, file_path: Path) -> Document:
        document_type = self._get_document_type(file_path)

        try:
            with Image.open(file_path) as image:
                image.load()

                extracted_text = self._run_ocr(image)

        except UnidentifiedImageError as error:
            raise OCRError(
                f"'{file_path.name}' is not a valid image file."
            ) from error

        except OSError as error:
            raise OCRError(
                f"Unable to read image '{file_path.name}'."
            ) from error

        except pytesseract.TesseractNotFoundError as error:
            raise OCRError(
                "Tesseract OCR is not installed or is not available "
                "on the system PATH."
            ) from error

        except pytesseract.TesseractError as error:
            raise OCRError(
                f"Tesseract OCR failed for '{file_path.name}': {error}"
            ) from error

        page = Page(
            number=1,
            text=extracted_text,
        )

        return Document(
            id=str(uuid4()),
            filename=file_path.name,
            filepath=file_path,
            document_type=document_type,
            pages=[page],
            metadata=DocumentMetadata(),
            file_size=file_path.stat().st_size,
            extracted_text=extracted_text,
            page_count=1,
        )

    @staticmethod
    def _run_ocr(image: Image.Image) -> str:
        """
        Run Tesseract OCR against an image.
        """

        text = pytesseract.image_to_string(
            image,
            lang="eng",
        )

        return text.strip()

    @staticmethod
    def _get_document_type(
        file_path: Path,
    ) -> DocumentType:
        """
        Convert an image extension into the domain document type.
        """

        extension = file_path.suffix.lower()

        document_types = {
            ".png": DocumentType.PNG,
            ".jpg": DocumentType.JPG,
            ".jpeg": DocumentType.JPEG,
        }

        return document_types.get(
            extension,
            DocumentType.UNKNOWN,
        )