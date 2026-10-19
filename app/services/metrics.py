"""Metrics and telemetry reporting for Notely dashboard and health checks."""

from typing import Any

from app.config import settings
from app.services.document_store import DocumentStore, get_document_store
from app.services.llm_provider import get_llm_provider
from app.services.vector_store import VectorStoreService, get_vector_store


class MetricsService:
    """Computes genuine system and usage metrics without fabrication."""

    def __init__(
        self,
        doc_store: DocumentStore | None = None,
        vector_store: VectorStoreService | None = None,
    ) -> None:
        self.doc_store = doc_store or get_document_store()
        self.vector_store = vector_store or get_vector_store()

    def get_system_metrics(self) -> dict[str, Any]:
        """Aggregate actual real-time application metrics."""
        docs = self.doc_store.list_documents()
        total_docs = len(docs)
        total_chunks = self.vector_store.get_chunk_count()
        total_pages = sum(d.page_count for d in docs)
        total_bytes = sum(d.file_size_bytes for d in docs)

        provider = get_llm_provider()

        recent_docs = [
            {
                "id": d.id,
                "name": d.name,
                "file_type": d.file_type,
                "created_at": d.created_at,
                "page_count": d.page_count,
                "chunk_count": d.chunk_count,
                "status": d.status,
            }
            for d in docs[:5]
        ]

        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_pages": total_pages,
            "total_storage_bytes": total_bytes,
            "embedding_model": settings.notely_embedding_model,
            "llm_provider": provider.provider_name,
            "is_mock_provider": provider.is_mock,
            "default_chunk_size": settings.notely_default_chunk_size,
            "default_chunk_overlap": settings.notely_default_chunk_overlap,
            "default_top_k": settings.notely_default_top_k,
            "recent_documents": recent_docs,
        }


def get_metrics_service() -> MetricsService:
    return MetricsService()
