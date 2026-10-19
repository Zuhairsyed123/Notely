"""Tests for document loader, text cleaner, and OCR/scanned detection."""

import pytest

from app.core.exceptions import (
    CorruptDocumentError,
    EmptyDocumentError,
    ScannedPDFError,
    UnsupportedFileTypeError,
)
from app.services.document_loader import DocumentLoader
from app.services.text_cleaner import clean_text, is_meaningful_text


def test_clean_text_normalization() -> None:
    raw = "Here is   some  text.\r\nNext line with soft-\nhyphenation and   spaces.\n\n\n\nParagraph 2."
    cleaned = clean_text(raw)
    assert "hyphenation" in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert is_meaningful_text(cleaned)


def test_load_valid_pdf(sample_pdf_bytes: bytes) -> None:
    doc = DocumentLoader.load_from_bytes("Lecture_1.pdf", sample_pdf_bytes)
    assert doc.name == "Lecture_1.pdf"
    assert doc.file_type == "pdf"
    assert doc.page_count == 2
    assert len(doc.pages) == 2
    assert "Photosynthesis" in doc.pages[0].cleaned_text
    assert "Cellular respiration" in doc.pages[1].cleaned_text
    assert doc.content_hash is not None


def test_load_valid_txt(sample_txt_bytes: bytes) -> None:
    doc = DocumentLoader.load_from_bytes("os_notes.txt", sample_txt_bytes)
    assert doc.name == "os_notes.txt"
    assert doc.file_type == "txt"
    assert doc.page_count == 1
    assert "Virtual Memory" in doc.pages[0].cleaned_text


def test_load_pasted_text() -> None:
    notes = "Algorithmic Complexity: Big O notation describes the upper bound of an algorithm."
    doc = DocumentLoader.load_from_pasted_text("CS101", notes)
    assert doc.name == "CS101"
    assert doc.file_type == "pasted"
    assert doc.page_count == 1
    assert "Big O notation" in doc.pages[0].cleaned_text


def test_scanned_pdf_detection(scanned_pdf_bytes: bytes) -> None:
    with pytest.raises(ScannedPDFError) as exc_info:
        DocumentLoader.load_from_bytes("scanned_doc.pdf", scanned_pdf_bytes)
    assert "OCR" in str(exc_info.value)


def test_empty_file_error() -> None:
    with pytest.raises(EmptyDocumentError):
        DocumentLoader.load_from_bytes("empty.txt", b"")


def test_corrupt_pdf_error() -> None:
    with pytest.raises(CorruptDocumentError):
        DocumentLoader.load_from_bytes("bad.pdf", b"Not a valid PDF header at all")


def test_unsupported_file_type() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        DocumentLoader.load_from_bytes("document.docx", b"dummy docx bytes")


def test_hash_calculation_consistency() -> None:
    text = "Consistent document content"
    hash1 = DocumentLoader.calculate_hash(text)
    hash2 = DocumentLoader.calculate_hash(text.encode("utf-8"))
    assert hash1 == hash2
    assert len(hash1) == 64
