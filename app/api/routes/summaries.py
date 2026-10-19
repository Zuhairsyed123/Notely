"""Summarization routes providing extractive and AI-based summary generation."""

from fastapi import APIRouter, HTTPException, status

from app.core.exceptions import DocumentNotFoundError
from app.schemas import SummarizeRequest, SummarizeResponse
from app.services.document_store import get_document_store
from app.services.summarizer import DocumentSummarizer

router = APIRouter(prefix="/documents", tags=["Summaries"])


@router.post("/{document_id}/summarize", response_model=SummarizeResponse)
def summarize_document(
    document_id: str,
    request: SummarizeRequest,
) -> SummarizeResponse:
    """Generate an extractive or provider-backed AI summary for a selected document."""
    doc_store = get_document_store()
    try:
        document = doc_store.get_document(document_id)
        summarizer = DocumentSummarizer()
        result = summarizer.summarize_document(
            document=document,
            mode=request.mode,
            max_sentences=request.max_sentences,
        )
        return SummarizeResponse(**result)
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {e}",
        )
