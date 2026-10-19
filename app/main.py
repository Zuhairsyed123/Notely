"""Main FastAPI application entry point for Notely."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chat, documents, health, study, summaries
from app.config import settings
from app.core.exceptions import (
    CorruptDocumentError,
    DocumentNotFoundError,
    EmptyDocumentError,
    InvalidChunkConfigError,
    NotelyError,
    ProviderError,
    ScannedPDFError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.services.metrics import get_metrics_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown hooks."""
    settings.ensure_directories()
    logger.info("Starting Notely API backend [Model=%s, Provider=%s]", settings.notely_embedding_model, settings.notely_llm_provider)
    yield
    logger.info("Shutting down Notely API backend.")


app = FastAPI(
    title="Notely API",
    description="Your notes. Smarter. Backend API for intelligent document ingestion, semantic RAG, and AI study tools.",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for local Streamlit and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(documents.search_router)
app.include_router(summaries.router)
app.include_router(chat.router)
app.include_router(study.router)


@app.get("/metrics", tags=["Metrics"])
def get_metrics() -> dict[str, Any]:
    """Retrieve comprehensive application telemetry for the dashboard."""
    metrics_svc = get_metrics_service()
    return metrics_svc.get_system_metrics()


# Global Exception Handlers
@app.exception_handler(DocumentNotFoundError)
async def handle_not_found(request: Request, exc: DocumentNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": exc.message, "type": "DocumentNotFoundError"},
    )


@app.exception_handler(ScannedPDFError)
async def handle_scanned_pdf(request: Request, exc: ScannedPDFError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.message, "type": "ScannedPDFError"},
    )


@app.exception_handler(EmptyDocumentError)
@app.exception_handler(InvalidChunkConfigError)
@app.exception_handler(UnsupportedFileTypeError)
@app.exception_handler(CorruptDocumentError)
async def handle_bad_request(request: Request, exc: NotelyError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.message, "type": exc.__class__.__name__},
    )


@app.exception_handler(ProviderError)
async def handle_provider_error(request: Request, exc: ProviderError):
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": exc.message, "type": "ProviderError"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
