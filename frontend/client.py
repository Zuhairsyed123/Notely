"""API Client for Notely Streamlit UI.

Provides seamless HTTP communication with the FastAPI backend, with a direct service-layer
fallback if running in unified local mode.
"""

from typing import Any

import requests

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class NotelyClient:
    """Client for Notely backend services."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or settings.api_base_url).rstrip("/")
        self._backend_available: bool | None = None

    def check_backend_alive(self) -> bool:
        """Check if FastAPI server is responsive on base_url."""
        try:
            res = requests.get(f"{self.base_url}/health", timeout=1.5)
            self._backend_available = res.status_code == 200
            return self._backend_available
        except Exception:
            self._backend_available = False
            return False

    def get_metrics(self) -> dict[str, Any]:
        """Fetch system-wide metrics and activity log."""
        if self.check_backend_alive():
            try:
                res = requests.get(f"{self.base_url}/metrics", timeout=5.0)
                if res.status_code == 200:
                    return res.json()
            except Exception as e:
                logger.warning("API metrics call failed: %s", e)

        # Fallback to direct service layer
        from app.services.metrics import get_metrics_service
        return get_metrics_service().get_system_metrics()

    def list_documents(self) -> list[dict[str, Any]]:
        """Fetch all indexed documents."""
        if self.check_backend_alive():
            try:
                res = requests.get(f"{self.base_url}/documents", timeout=5.0)
                if res.status_code == 200:
                    return res.json().get("documents", [])
            except Exception as e:
                logger.warning("API list_documents call failed: %s", e)

        from app.services.document_store import get_document_store
        docs = get_document_store().list_documents()
        return [
            {
                "id": d.id,
                "name": d.name,
                "file_type": d.file_type,
                "file_size_bytes": d.file_size_bytes,
                "page_count": d.page_count,
                "chunk_count": d.chunk_count,
                "status": d.status,
                "created_at": d.created_at,
                "error_message": d.error_message,
            }
            for d in docs
        ]

    def get_document(self, document_id: str) -> dict[str, Any]:
        """Fetch details of a single document."""
        if self.check_backend_alive():
            try:
                res = requests.get(f"{self.base_url}/documents/{document_id}", timeout=5.0)
                if res.status_code == 200:
                    return res.json()
            except Exception as e:
                logger.warning("API get_document call failed: %s", e)

        from app.services.document_store import get_document_store
        doc = get_document_store().get_document(document_id)
        return {
            "document": {
                "id": doc.id,
                "name": doc.name,
                "file_type": doc.file_type,
                "file_size_bytes": doc.file_size_bytes,
                "page_count": doc.page_count,
                "chunk_count": doc.chunk_count,
                "status": doc.status,
                "created_at": doc.created_at,
            },
            "full_text_preview": doc.get_full_text()[:1000],
            "pages": [
                {
                    "page_number": p.page_number,
                    "char_count": p.char_count,
                    "preview": p.cleaned_text[:300] + "...",
                }
                for p in doc.pages
            ],
        }

    def delete_document(self, document_id: str) -> bool:
        """Delete document from store and vector database."""
        if self.check_backend_alive():
            try:
                res = requests.delete(f"{self.base_url}/documents/{document_id}", timeout=5.0)
                return res.status_code == 200
            except Exception as e:
                logger.warning("API delete_document call failed: %s", e)

        from app.services.document_store import get_document_store
        from app.services.vector_store import get_vector_store
        return get_document_store().delete_document(document_id, get_vector_store())

    def upload_file(
        self,
        filename: str,
        content: bytes,
        chunk_size: int,
        chunk_overlap: int,
    ) -> dict[str, Any]:
        """Upload a PDF or TXT file."""
        if self.check_backend_alive():
            try:
                files = {"file": (filename, content)}
                data = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
                res = requests.post(f"{self.base_url}/documents/upload", files=files, data=data, timeout=30.0)
                if res.status_code in (200, 201):
                    return res.json()
                detail = res.json().get("detail", res.text)
                raise Exception(detail)
            except Exception as e:
                if self._backend_available:
                    raise e

        # Fallback to direct service call
        from app.services.chunker import DocumentChunker
        from app.services.document_loader import DocumentLoader
        from app.services.document_store import get_document_store
        from app.services.vector_store import get_vector_store

        doc_store = get_document_store()
        doc = DocumentLoader.load_from_bytes(
            filename=filename,
            content=content,
            existing_hashes=doc_store.get_existing_hashes(),
        )
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_document(doc)
        v_store = get_vector_store()
        indexed = v_store.add_chunks(chunks)
        doc.chunk_count = indexed
        doc.status = "indexed"
        doc_store.save_document(doc)
        return {
            "id": doc.id,
            "name": doc.name,
            "file_type": doc.file_type,
            "file_size_bytes": doc.file_size_bytes,
            "page_count": doc.page_count,
            "chunk_count": doc.chunk_count,
            "status": doc.status,
            "created_at": doc.created_at,
        }

    def paste_notes(
        self,
        title: str,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> dict[str, Any]:
        """Ingest pasted raw text notes."""
        if self.check_backend_alive():
            try:
                payload = {"title": title, "text": text}
                params = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
                res = requests.post(
                    f"{self.base_url}/documents/paste", json=payload, params=params, timeout=60.0
                )
                if res.status_code in (200, 201):
                    return res.json()
                detail = res.json().get("detail", res.text)
                raise Exception(detail)
            except Exception as e:
                if self._backend_available:
                    raise e

        from app.services.chunker import DocumentChunker
        from app.services.document_loader import DocumentLoader
        from app.services.document_store import get_document_store
        from app.services.vector_store import get_vector_store

        doc_store = get_document_store()
        doc = DocumentLoader.load_from_pasted_text(
            title=title,
            text=text,
            existing_hashes=doc_store.get_existing_hashes(),
        )
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_document(doc)
        v_store = get_vector_store()
        indexed = v_store.add_chunks(chunks)
        doc.chunk_count = indexed
        doc.status = "indexed"
        doc_store.save_document(doc)
        return {
            "id": doc.id,
            "name": doc.name,
            "file_type": doc.file_type,
            "file_size_bytes": doc.file_size_bytes,
            "page_count": doc.page_count,
            "chunk_count": doc.chunk_count,
            "status": doc.status,
            "created_at": doc.created_at,
        }

    def inspect_chunks(
        self,
        document_id: str,
        chunk_size: int,
        chunk_overlap: int,
        reindex: bool = False,
    ) -> dict[str, Any]:
        """Simulate chunking or re-index with custom parameters."""
        if self.check_backend_alive():
            try:
                payload = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
                params = {"reindex": reindex}
                res = requests.post(
                    f"{self.base_url}/documents/{document_id}/chunks",
                    json=payload,
                    params=params,
                    timeout=20.0,
                )
                if res.status_code == 200:
                    return res.json()
                raise Exception(res.json().get("detail", res.text))
            except Exception as e:
                if self._backend_available:
                    raise e

        from app.services.chunker import DocumentChunker
        from app.services.document_store import get_document_store
        from app.services.vector_store import get_vector_store

        doc_store = get_document_store()
        doc = doc_store.get_document(document_id)
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_document(doc)

        if reindex:
            v_store = get_vector_store()
            v_store.delete_document_chunks(document_id)
            v_store.add_chunks(chunks)
            doc.chunk_count = len(chunks)
            doc_store.save_document(doc)

        return {
            "document_id": doc.id,
            "document_name": doc.name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "chunk_id": c.id,
                    "document_id": c.document_id,
                    "document_name": c.document_name,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                    "text": c.text,
                    "char_count": c.char_count,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                }
                for c in chunks
            ],
        }

    def summarize_document(
        self,
        document_id: str,
        mode: str,
        max_sentences: int = 5,
    ) -> dict[str, Any]:
        """Summarize document using extractive or AI modes."""
        if self.check_backend_alive():
            try:
                payload = {"mode": mode, "max_sentences": max_sentences}
                res = requests.post(
                    f"{self.base_url}/documents/{document_id}/summarize",
                    json=payload,
                    timeout=45.0,
                )
                if res.status_code == 200:
                    return res.json()
                raise Exception(res.json().get("detail", res.text))
            except Exception as e:
                if self._backend_available:
                    raise e

        from app.services.document_store import get_document_store
        from app.services.summarizer import DocumentSummarizer

        doc = get_document_store().get_document(document_id)
        summarizer = DocumentSummarizer()
        return summarizer.summarize_document(doc, mode=mode, max_sentences=max_sentences)

    def chat_with_notes(
        self,
        query: str,
        document_id: str | None = None,
        top_k: int = 4,
    ) -> dict[str, Any]:
        """Ask question via RAG pipeline."""
        if self.check_backend_alive():
            try:
                payload = {"query": query, "document_id": document_id, "top_k": top_k}
                res = requests.post(f"{self.base_url}/chat", json=payload, timeout=30.0)
                if res.status_code == 200:
                    return res.json()
                raise Exception(res.json().get("detail", res.text))
            except Exception as e:
                if self._backend_available:
                    raise e

        from app.services.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()
        return pipeline.answer_question(query=query, document_id=document_id, top_k=top_k)

    def search_documents(
        self,
        query: str,
        document_id: str | None = None,
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        """Semantic search query across indexed chunks."""
        if self.check_backend_alive():
            try:
                payload = {"query": query, "document_id": document_id, "top_k": top_k}
                res = requests.post(f"{self.base_url}/search", json=payload, timeout=15.0)
                if res.status_code == 200:
                    return res.json().get("results", [])
            except Exception as e:
                if self._backend_available:
                    raise e

        from app.services.retriever import SemanticRetriever

        retriever = SemanticRetriever()
        return retriever.retrieve(query=query, top_k=top_k, document_id=document_id)

    def generate_flashcards(self, document_id: str | None = None, count: int = 5) -> dict[str, Any]:
        """Generate study flashcards."""
        if self.check_backend_alive():
            try:
                payload = {"document_id": document_id, "count": count}
                res = requests.post(f"{self.base_url}/study/flashcards", json=payload, timeout=30.0)
                if res.status_code == 200:
                    return res.json()
                raise Exception(res.json().get("detail", res.text))
            except Exception as e:
                if self._backend_available:
                    raise e

        from app.services.study_generator import StudyGenerator

        generator = StudyGenerator()
        return generator.generate_flashcards(document_id=document_id, count=count)

    def generate_quiz(self, document_id: str | None = None, count: int = 5) -> dict[str, Any]:
        """Generate multiple choice quiz."""
        if self.check_backend_alive():
            try:
                payload = {"document_id": document_id, "question_count": count}
                res = requests.post(f"{self.base_url}/study/quiz", json=payload, timeout=30.0)
                if res.status_code == 200:
                    return res.json()
                raise Exception(res.json().get("detail", res.text))
            except Exception as e:
                if self._backend_available:
                    raise e

        from app.services.study_generator import StudyGenerator

        generator = StudyGenerator()
        return generator.generate_quiz(document_id=document_id, question_count=count)

    def generate_revision_notes(self, document_id: str | None = None) -> dict[str, Any]:
        """Generate structured revision sheets."""
        if self.check_backend_alive():
            try:
                payload = {"document_id": document_id}
                res = requests.post(f"{self.base_url}/study/revision", json=payload, timeout=30.0)
                if res.status_code == 200:
                    return res.json()
                raise Exception(res.json().get("detail", res.text))
            except Exception as e:
                if self._backend_available:
                    raise e

        from app.services.study_generator import StudyGenerator

        generator = StudyGenerator()
        return generator.generate_revision_notes(document_id=document_id)


# Singleton instance
client = NotelyClient()
