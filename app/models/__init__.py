"""Domain data models for Notely."""

from app.models.chunk import DocumentChunk
from app.models.document import Document, DocumentPage

__all__ = ["Document", "DocumentPage", "DocumentChunk"]
