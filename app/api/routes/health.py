"""Health check endpoint providing diagnostic telemetry."""

from fastapi import APIRouter

from app.config import settings
from app.schemas import HealthResponse
from app.services.document_store import get_document_store
from app.services.llm_provider import get_llm_provider
from app.services.vector_store import get_vector_store

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return system readiness, active providers, and indexing telemetry."""
    doc_store = get_document_store()
    vector_store = get_vector_store()
    provider = get_llm_provider()

    return HealthResponse(
        status="ok",
        version="0.1.0",
        embedding_model=settings.notely_embedding_model,
        llm_provider=provider.provider_name,
        document_count=len(doc_store.list_documents()),
        chunk_count=vector_store.get_chunk_count(),
    )
