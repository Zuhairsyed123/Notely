"""Persistent metadata store for ingested documents surviving application restarts."""

import json
from pathlib import Path
from threading import Lock

from app.config import settings
from app.core.exceptions import DocumentNotFoundError
from app.core.logging import get_logger
from app.models.document import Document
from app.services.vector_store import VectorStoreService, get_vector_store

logger = get_logger(__name__)


class DocumentStore:
    """Manages thread-safe local JSON persistence for Document metadata records."""

    def __init__(self, metadata_path: Path | None = None) -> None:
        self.metadata_path = metadata_path or settings.metadata_file
        self._lock = Lock()
        self._docs: dict[str, Document] = {}
        self._load()

    def _load(self) -> None:
        """Load document records from persistent metadata JSON."""
        if not self.metadata_path.exists():
            self._docs = {}
            return

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._docs = {
                    doc_id: Document.model_validate(doc_dict)
                    for doc_id, doc_dict in data.items()
                }
            logger.info("Loaded %d documents from metadata store", len(self._docs))
        except Exception as e:
            logger.error("Failed to load metadata file at %s: %s", self.metadata_path, str(e))
            self._docs = {}

    def _persist(self) -> None:
        """Serialize current documents dictionary to atomic JSON file."""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.metadata_path.with_suffix(".tmp")
        try:
            serializable = {
                doc_id: doc.model_dump() for doc_id, doc in self._docs.items()
            }
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
            temp_file.replace(self.metadata_path)
        except Exception as e:
            logger.error("Failed to write document metadata: %s", str(e))
            if temp_file.exists():
                temp_file.unlink()

    def save_document(self, document: Document) -> None:
        """Add or update a document in the store."""
        with self._lock:
            self._docs[document.id] = document
            self._persist()

    def get_document(self, doc_id: str) -> Document:
        """Fetch a document by ID or raise DocumentNotFoundError."""
        with self._lock:
            doc = self._docs.get(doc_id)
            if not doc:
                raise DocumentNotFoundError(f"Document with ID '{doc_id}' not found.")
            return doc

    def list_documents(self) -> list[Document]:
        """Return all documents ordered by creation date descending."""
        with self._lock:
            docs = list(self._docs.values())
            docs.sort(key=lambda d: d.created_at, reverse=True)
            return docs

    def delete_document(
        self, doc_id: str, vector_store: VectorStoreService | None = None
    ) -> bool:
        """Remove document record, its persisted chunks, and any raw uploaded artifacts."""
        with self._lock:
            if doc_id not in self._docs:
                raise DocumentNotFoundError(f"Document with ID '{doc_id}' not found.")
            del self._docs[doc_id]
            self._persist()

        # Delete associated vector embeddings in ChromaDB
        v_store = vector_store or get_vector_store()
        deleted_chunks = v_store.delete_document_chunks(doc_id)
        logger.info("Deleted document %s and %d associated vector chunks", doc_id, deleted_chunks)
        return True

    def get_existing_hashes(self) -> set[str]:
        """Get set of content hashes of all stored documents to prevent duplicates."""
        with self._lock:
            return {doc.content_hash for doc in self._docs.values() if doc.content_hash}


# Singleton document store
_document_store: DocumentStore | None = None


def get_document_store() -> DocumentStore:
    """Retrieve global DocumentStore instance."""
    global _document_store
    if _document_store is None:
        _document_store = DocumentStore()
    return _document_store
