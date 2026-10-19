"""Document domain models representing ingested notes and materials."""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """Represents a single page or section of an ingested document."""

    page_number: int = Field(..., description="1-based page number")
    raw_text: str = Field(..., description="Original raw text extracted from page")
    cleaned_text: str = Field(..., description="Whitespace-normalized cleaned text")
    char_count: int = Field(default=0, description="Length of cleaned text in characters")


class Document(BaseModel):
    """Complete document record including page texts and processing state."""

    id: str = Field(..., description="Unique document ID (UUID)")
    name: str = Field(..., description="Original filename or user title")
    file_type: Literal["pdf", "txt", "pasted"] = Field(..., description="Source format")
    content_hash: str = Field(..., description="SHA-256 hash of extracted text")
    file_size_bytes: int = Field(default=0, description="Size of source file or string in bytes")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 creation timestamp",
    )
    page_count: int = Field(default=1, description="Number of pages in the document")
    chunk_count: int = Field(default=0, description="Total chunks produced during processing")
    status: Literal["uploaded", "chunked", "indexed", "failed"] = Field(
        default="uploaded", description="Processing lifecycle state"
    )
    error_message: str | None = Field(default=None, description="Diagnostic error message if failed")
    pages: list[DocumentPage] = Field(default_factory=list, description="Extracted document pages")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata tags")

    def get_full_text(self) -> str:
        """Concatenate all cleaned page texts with page demarcation."""
        return "\n\n".join(
            f"[Page {p.page_number}]\n{p.cleaned_text}" for p in self.pages if p.cleaned_text.strip()
        )
