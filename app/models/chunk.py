"""Chunk domain model representing an indexed text segment."""

from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a discrete semantic chunk of a document with full provenance metadata."""

    id: str = Field(..., description="Deterministic unique identifier for the chunk")
    document_id: str = Field(..., description="Parent document identifier")
    document_name: str = Field(..., description="Parent document filename or title")
    chunk_index: int = Field(..., description="Zero-based sequence index within the document")
    page_number: int = Field(..., description="1-based page number where the chunk originated")
    text: str = Field(..., description="Normalized text content of the chunk")
    char_count: int = Field(..., description="Character count of the chunk text")
    start_char: int = Field(default=0, description="Start character offset within page or doc")
    end_char: int = Field(default=0, description="End character offset within page or doc")
    extra_metadata: dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")

    def to_metadata_dict(self) -> dict[str, Any]:
        """Convert chunk fields into ChromaDB compatible metadata dictionary (scalar values only)."""
        return {
            "chunk_id": self.id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "char_count": self.char_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }
