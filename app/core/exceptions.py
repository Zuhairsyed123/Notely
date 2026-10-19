"""Custom exception classes for Notely."""


class NotelyError(Exception):
    """Base exception for all Notely application errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DocumentNotFoundError(NotelyError):
    """Raised when a requested document ID does not exist."""
    pass


class EmptyDocumentError(NotelyError):
    """Raised when an uploaded or pasted document has no content."""
    pass


class ScannedPDFError(NotelyError):
    """Raised when a PDF contains only scanned images with no extractable text layer."""
    pass


class CorruptDocumentError(NotelyError):
    """Raised when a file cannot be parsed or decoded."""
    pass


class UnsupportedFileTypeError(NotelyError):
    """Raised when an uploaded file is not supported."""
    pass


class FileTooLargeError(NotelyError):
    """Raised when an uploaded file exceeds the maximum allowed size."""
    pass


class InvalidChunkConfigError(NotelyError):
    """Raised when chunk size or overlap parameters are invalid."""
    pass


class EmbeddingModelError(NotelyError):
    """Raised when the embedding model fails to initialize or generate vectors."""
    pass


class ProviderError(NotelyError):
    """Raised when an LLM provider encounters a configuration or execution error."""
    pass
