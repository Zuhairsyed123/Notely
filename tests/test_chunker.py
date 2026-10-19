"""Tests for intelligent document chunking engine."""

import pytest

from app.core.exceptions import InvalidChunkConfigError
from app.models.document import Document, DocumentPage
from app.services.chunker import DocumentChunker


def test_chunk_validation_errors() -> None:
    # Size <= 0
    with pytest.raises(InvalidChunkConfigError):
        DocumentChunker(chunk_size=0, chunk_overlap=0)

    # Overlap < 0
    with pytest.raises(InvalidChunkConfigError):
        DocumentChunker(chunk_size=100, chunk_overlap=-10)

    # Overlap >= Size
    with pytest.raises(InvalidChunkConfigError):
        DocumentChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(InvalidChunkConfigError):
        DocumentChunker(chunk_size=100, chunk_overlap=150)


def test_chunk_metadata_and_deterministic_ids(sample_document: Document) -> None:
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_document(sample_document)

    assert len(chunks) > 0
    for idx, c in enumerate(chunks):
        assert c.id == f"{sample_document.id}_chunk_{idx:04d}"
        assert c.document_id == sample_document.id
        assert c.document_name == sample_document.name
        assert c.chunk_index == idx
        assert c.char_count == len(c.text)
        assert c.page_number in (1, 2)


def test_chunking_long_paragraphs() -> None:
    # Construct doc with massive unbroken paragraph
    long_para = "Word " * 300  # ~1500 chars
    doc = Document(
        id="long_doc",
        name="long.txt",
        file_type="txt",
        content_hash="hash1",
        pages=[DocumentPage(page_number=1, raw_text=long_para, cleaned_text=long_para, char_count=len(long_para))],
    )

    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for c in chunks:
        # Each chunk should respect max size reasonably (within word boundary)
        assert len(c.text) <= 250
        assert c.id.startswith("long_doc_chunk_")
