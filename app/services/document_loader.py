"""Document loader and validator supporting PDF, TXT, and pasted notes."""

import hashlib
import re
import uuid
from pathlib import Path

import pymupdf

from app.config import settings
from app.core.exceptions import (
    CorruptDocumentError,
    EmptyDocumentError,
    FileTooLargeError,
    ScannedPDFError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.models.document import Document, DocumentPage

logger = get_logger(__name__)


def clean_text(raw_text: str) -> str:
    """Normalize whitespace and formatting without destroying paragraphs, headings, or lists."""
    if not raw_text:
        return ""

    # Normalize line endings
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize non-standard spaces
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("\t", "    ")

    # Fix hyphenated line breaks
    text = re.sub(r"(\b[a-zA-Z]+)-\n\s*([a-zA-Z]+\b)", r"\1\2", text)

    # Process line-by-line
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        leading_spaces = len(line) - len(line.lstrip(" "))
        content = re.sub(r"[ ]{2,}", " ", line.strip())
        if content:
            if leading_spaces >= 2 and (content.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5."))):
                cleaned_lines.append("  " + content)
            else:
                cleaned_lines.append(content)
        else:
            cleaned_lines.append("")

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_meaningful_text(text: str, min_chars: int = 15) -> bool:
    """Check if the provided string contains meaningful alphanumeric content."""
    if not text:
        return False
    alphanumeric_count = sum(1 for c in text if c.isalnum())
    return alphanumeric_count >= min_chars


class DocumentLoader:
    """Handles parsing, validation, sanitization, and structured extraction of documents."""

    @staticmethod
    def calculate_hash(content: bytes | str) -> str:
        """Compute SHA-256 hash for deduplication checking."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    @classmethod
    def load_from_bytes(
        cls,
        filename: str,
        content: bytes,
        existing_hashes: set[str] | None = None,
    ) -> Document:
        """Validate, extract, and clean text from raw file bytes (PDF or TXT)."""
        # 1. Size validation
        file_size = len(content)
        if file_size == 0:
            raise EmptyDocumentError(f"Uploaded file '{filename}' is completely empty (0 bytes).")
        if file_size > settings.max_upload_size_bytes:
            max_mb = settings.max_upload_size_bytes / (1024 * 1024)
            raise FileTooLargeError(
                f"File size ({file_size / (1024 * 1024):.1f} MB) exceeds maximum limit of {max_mb:.0f} MB."
            )

        # 2. File extension validation
        extension = Path(filename).suffix.lower()
        if extension not in [".pdf", ".txt"]:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{extension}'. Notely currently supports PDF and TXT files."
            )

        # 3. Duplicate detection
        content_hash = cls.calculate_hash(content)
        if existing_hashes and content_hash in existing_hashes:
            logger.info("Duplicate document detected with hash: %s", content_hash[:8])

        doc_id = uuid.uuid4().hex

        # 4. Extraction based on type
        if extension == ".pdf":
            return cls._extract_pdf(doc_id, filename, content, content_hash, file_size)
        else:
            return cls._extract_txt(doc_id, filename, content, content_hash, file_size)

    @classmethod
    def load_from_pasted_text(
        cls,
        title: str,
        text: str,
        existing_hashes: set[str] | None = None,
    ) -> Document:
        """Process manually pasted user notes into a valid Document record."""
        raw_text = text.strip()
        if not raw_text:
            raise EmptyDocumentError("Pasted notes cannot be empty.")

        cleaned = clean_text(raw_text)
        if not is_meaningful_text(cleaned, min_chars=5):
            raise EmptyDocumentError("Pasted notes do not contain sufficient readable text.")

        content_hash = cls.calculate_hash(cleaned)
        doc_id = uuid.uuid4().hex

        page = DocumentPage(
            page_number=1,
            raw_text=raw_text,
            cleaned_text=cleaned,
            char_count=len(cleaned),
        )

        doc = Document(
            id=doc_id,
            name=title.strip() or f"Pasted Note {doc_id[:6]}",
            file_type="pasted",
            content_hash=content_hash,
            file_size_bytes=len(raw_text.encode("utf-8")),
            page_count=1,
            chunk_count=0,
            status="uploaded",
            pages=[page],
            metadata={"source": "pasted_text"},
        )
        return doc

    @classmethod
    def _extract_pdf(
        cls,
        doc_id: str,
        filename: str,
        content: bytes,
        content_hash: str,
        file_size: int,
    ) -> Document:
        """Extract text from PDF page by page with scanned/OCR detection."""
        try:
            pdf_doc = pymupdf.open(stream=content, filetype="pdf")
        except Exception as e:
            logger.error("Failed to parse PDF %s: %s", filename, str(e))
            raise CorruptDocumentError(
                f"Unable to read PDF file '{filename}'. The file may be corrupt or encrypted."
            ) from e

        total_pages = len(pdf_doc)
        if total_pages == 0:
            pdf_doc.close()
            raise EmptyDocumentError(f"PDF document '{filename}' contains 0 pages.")

        pages: list[DocumentPage] = []
        total_extracted_chars = 0
        has_images = False

        for page_idx in range(total_pages):
            page = pdf_doc[page_idx]
            page_number = page_idx + 1
            raw_page_text = page.get_text("text") or ""
            cleaned_page_text = clean_text(raw_page_text)

            char_count = len(cleaned_page_text)
            total_extracted_chars += char_count

            if not has_images and len(page.get_images()) > 0:
                has_images = True

            pages.append(
                DocumentPage(
                    page_number=page_number,
                    raw_text=raw_page_text,
                    cleaned_text=cleaned_page_text,
                    char_count=char_count,
                )
            )

        pdf_doc.close()

        # Check for scanned or textless PDF
        if total_extracted_chars < 30:
            if has_images or total_pages > 0:
                raise ScannedPDFError(
                    f"PDF '{filename}' appears to be a scanned document or image without an embedded "
                    "text layer. Optical Character Recognition (OCR) is required before this file "
                    "can be indexed."
                )
            else:
                raise EmptyDocumentError(f"PDF '{filename}' contains no extractable text.")

        doc = Document(
            id=doc_id,
            name=filename,
            file_type="pdf",
            content_hash=content_hash,
            file_size_bytes=file_size,
            page_count=total_pages,
            chunk_count=0,
            status="uploaded",
            pages=pages,
            metadata={"total_pages": total_pages},
        )
        return doc

    @classmethod
    def _extract_txt(
        cls,
        doc_id: str,
        filename: str,
        content: bytes,
        content_hash: str,
        file_size: int,
    ) -> Document:
        """Decode and extract text from a plain TXT file."""
        # Try UTF-8 first, fallback to latin-1 / cp1252
        raw_text = None
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                raw_text = content.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if raw_text is None:
            raise CorruptDocumentError(
                f"Failed to decode TXT file '{filename}' with supported text encodings."
            )

        cleaned = clean_text(raw_text)
        if not is_meaningful_text(cleaned, min_chars=10):
            raise EmptyDocumentError(f"File '{filename}' contains no meaningful text content.")

        page = DocumentPage(
            page_number=1,
            raw_text=raw_text,
            cleaned_text=cleaned,
            char_count=len(cleaned),
        )

        doc = Document(
            id=doc_id,
            name=filename,
            file_type="txt",
            content_hash=content_hash,
            file_size_bytes=file_size,
            page_count=1,
            chunk_count=0,
            status="uploaded",
            pages=[page],
            metadata={},
        )
        return doc
