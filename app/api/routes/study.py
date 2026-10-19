"""Study tools routes: flashcards, multiple-choice quizzes, and structured revision guides."""

from fastapi import APIRouter, HTTPException, status

from app.core.exceptions import NotelyError
from app.schemas import (
    FlashcardsRequest,
    FlashcardsResponse,
    QuizRequest,
    QuizResponse,
    RevisionNotesRequest,
    RevisionNotesResponse,
)
from app.services.study_generator import StudyGenerator

router = APIRouter(prefix="/study", tags=["Study Tools"])


@router.post("/flashcards", response_model=FlashcardsResponse)
def generate_flashcards(request: FlashcardsRequest) -> FlashcardsResponse:
    """Generate study flashcards grounded in indexed material."""
    try:
        generator = StudyGenerator()
        result = generator.generate_flashcards(
            document_id=request.document_id,
            count=request.count,
        )
        return FlashcardsResponse(**result)
    except NotelyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Flashcard generation failed: {e}",
        )


@router.post("/quiz", response_model=QuizResponse)
def generate_quiz(request: QuizRequest) -> QuizResponse:
    """Generate grounded multiple-choice questions with answers and source explanations."""
    try:
        generator = StudyGenerator()
        result = generator.generate_quiz(
            document_id=request.document_id,
            question_count=request.question_count,
        )
        return QuizResponse(**result)
    except NotelyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quiz generation failed: {e}",
        )


@router.post("/revision", response_model=RevisionNotesResponse)
def generate_revision_notes(request: RevisionNotesRequest) -> RevisionNotesResponse:
    """Generate structured revision sheets with definitions, concepts, and formulas."""
    try:
        generator = StudyGenerator()
        result = generator.generate_revision_notes(
            document_id=request.document_id,
        )
        return RevisionNotesResponse(**result)
    except NotelyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Revision notes generation failed: {e}",
        )
