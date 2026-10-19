"""Intelligent chunking engine for Notely.

Provides paragraph- and sentence-aware sliding-window chunking with provenance tracking.
"""

import re

from app.core.exceptions import InvalidChunkConfigError
from app.core.logging import get_logger
from app.models.chunk import DocumentChunk
from app.models.document import Document

logger = get_logger(__name__)


class DocumentChunker:
    """Paragraph- and sentence-aware sliding window chunker."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._validate_config(chunk_size, chunk_overlap)

    @staticmethod
    def _validate_config(chunk_size: int, chunk_overlap: int) -> None:
        """Ensure chunk size is positive and overlap is non-negative and strictly less than size."""
        if chunk_size <= 0:
            raise InvalidChunkConfigError(f"Chunk size must be positive, got {chunk_size}.")
        if chunk_overlap < 0:
            raise InvalidChunkConfigError(f"Chunk overlap cannot be negative, got {chunk_overlap}.")
        if chunk_overlap >= chunk_size:
            raise InvalidChunkConfigError(
                f"Chunk overlap ({chunk_overlap}) must be strictly smaller than chunk size ({chunk_size})."
            )

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Split text into sentences using regex boundary matching while preserving content."""
        # Match punctuation followed by whitespace and uppercase or end of string
        pattern = r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])"
        parts = re.split(pattern, text)
        sentences = [p.strip() for p in parts if p.strip()]
        return sentences if sentences else [text.strip()]

    def _split_into_atomic_units(self, text: str) -> list[str]:
        """Break text into paragraphs, then sentences if paragraphs are oversized."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        units: list[str] = []

        for para in paragraphs:
            if len(para) <= self.chunk_size:
                units.append(para)
            else:
                # Oversized paragraph: split into sentences
                sentences = self._split_into_sentences(para)
                for sentence in sentences:
                    if len(sentence) <= self.chunk_size:
                        units.append(sentence)
                    else:
                        # Oversized sentence: hard-break by words safely
                        words = sentence.split(" ")
                        curr_part: list[str] = []
                        curr_len = 0
                        for word in words:
                            if curr_len + len(word) + 1 > self.chunk_size and curr_part:
                                units.append(" ".join(curr_part))
                                curr_part = [word]
                                curr_len = len(word)
                            else:
                                curr_part.append(word)
                                curr_len += len(word) + 1
                        if curr_part:
                            units.append(" ".join(curr_part))
        return units

    def chunk_document(self, document: Document) -> list[DocumentChunk]:
        """Process a Document's pages into an ordered list of deterministic DocumentChunks."""
        chunks: list[DocumentChunk] = []
        global_chunk_idx = 0

        for page in document.pages:
            cleaned_text = page.cleaned_text.strip()
            if not cleaned_text:
                continue

            atomic_units = self._split_into_atomic_units(cleaned_text)
            if not atomic_units:
                continue

            current_units: list[str] = []
            current_len = 0
            page_char_offset = 0

            for unit in atomic_units:
                unit_len = len(unit)
                # If adding this unit exceeds chunk_size and we already have units, emit chunk
                if current_units and (current_len + unit_len + 2 > self.chunk_size):
                    chunk_text = "\n\n".join(current_units)
                    chunk_id = f"{document.id}_chunk_{global_chunk_idx:04d}"

                    start_char = page_char_offset
                    end_char = start_char + len(chunk_text)

                    chunks.append(
                        DocumentChunk(
                            id=chunk_id,
                            document_id=document.id,
                            document_name=document.name,
                            chunk_index=global_chunk_idx,
                            page_number=page.page_number,
                            text=chunk_text,
                            char_count=len(chunk_text),
                            start_char=start_char,
                            end_char=end_char,
                            extra_metadata={
                                "chunk_size": self.chunk_size,
                                "chunk_overlap": self.chunk_overlap,
                            },
                        )
                    )
                    global_chunk_idx += 1
                    page_char_offset = max(0, end_char - self.chunk_overlap)

                    # Carry over overlap
                    # Pick units from the end of current_units that fit within chunk_overlap
                    overlap_units: list[str] = []
                    overlap_len = 0
                    for u in reversed(current_units):
                        if overlap_len + len(u) + 2 <= self.chunk_overlap:
                            overlap_units.insert(0, u)
                            overlap_len += len(u) + 2
                        else:
                            break

                    current_units = overlap_units
                    current_len = overlap_len

                current_units.append(unit)
                current_len += unit_len + 2

            # Flush remaining buffer for this page
            if current_units:
                chunk_text = "\n\n".join(current_units).strip()
                if chunk_text:
                    chunk_id = f"{document.id}_chunk_{global_chunk_idx:04d}"
                    start_char = page_char_offset
                    end_char = start_char + len(chunk_text)

                    chunks.append(
                        DocumentChunk(
                            id=chunk_id,
                            document_id=document.id,
                            document_name=document.name,
                            chunk_index=global_chunk_idx,
                            page_number=page.page_number,
                            text=chunk_text,
                            char_count=len(chunk_text),
                            start_char=start_char,
                            end_char=end_char,
                            extra_metadata={
                                "chunk_size": self.chunk_size,
                                "chunk_overlap": self.chunk_overlap,
                            },
                        )
                    )
                    global_chunk_idx += 1

        logger.info(
            "Chunked document '%s' (%s) into %d chunks [size=%d, overlap=%d]",
            document.name,
            document.id[:8],
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks
