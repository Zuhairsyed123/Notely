"""Chat endpoint for evidence-grounded RAG question answering."""

from fastapi import APIRouter, HTTPException, status

from app.core.exceptions import NotelyError
from app.schemas import ChatRequest, ChatResponse
from app.services.rag_pipeline import RAGPipeline

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
def chat_with_notes(request: ChatRequest) -> ChatResponse:
    """Ask questions grounded in indexed notes with verified citations and evidence excerpts."""
    try:
        pipeline = RAGPipeline()
        result = pipeline.answer_question(
            query=request.query,
            document_id=request.document_id,
            top_k=request.top_k,
        )
        return ChatResponse(**result)
    except NotelyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question answering failed: {e}",
        )
