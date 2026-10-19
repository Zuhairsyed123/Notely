"""Study tools generator for flashcards, multiple-choice quizzes, and structured revision notes."""

from typing import Any

from app.core.exceptions import NotelyError
from app.core.logging import get_logger
from app.services.llm_provider import BaseLLMProvider, get_llm_provider
from app.services.vector_store import VectorStoreService, get_vector_store

logger = get_logger(__name__)


class StudyGenerator:
    """Orchestrates generation of flashcards, quizzes, and revision guides from indexed chunks."""

    def __init__(
        self,
        vector_store: VectorStoreService | None = None,
        provider: BaseLLMProvider | None = None,
    ) -> None:
        self.vector_store = vector_store or get_vector_store()
        self.provider = provider or get_llm_provider()

    def _get_context_chunks(
        self, document_id: str | None = None, max_chunks: int = 12
    ) -> list[dict[str, Any]]:
        """Retrieve a representative sample of chunks from the document or collection."""
        if document_id:
            chunks = self.vector_store.get_chunks_for_document(document_id)
        else:
            # Query broad conceptual terms across the collection
            chunks = self.vector_store.query_similarity(
                query="key concepts definitions principles theory application overview",
                top_k=max_chunks,
            )

        if not chunks:
            # Try getting any chunks if semantic query yielded nothing
            if document_id:
                chunks = self.vector_store.get_chunks_for_document(document_id)
        return chunks[:max_chunks]

    def generate_flashcards(
        self,
        document_id: str | None = None,
        count: int = 5,
    ) -> dict[str, Any]:
        """Generate interactive flashcards grounded in document chunks."""
        chunks = self._get_context_chunks(document_id, max_chunks=10)
        if not chunks:
            raise NotelyError(
                "No indexed study material available to generate flashcards. Please upload or index a document first."
            )

        cards = self.provider.generate_flashcards(chunks, count=count)
        return {
            "document_id": document_id,
            "total_cards": len(cards),
            "is_ai_generated": not self.provider.is_mock,
            "provider": self.provider.provider_name,
            "flashcards": cards,
        }

    def generate_quiz(
        self,
        document_id: str | None = None,
        question_count: int = 5,
    ) -> dict[str, Any]:
        """Generate multiple-choice quiz questions with verified explanations."""
        chunks = self._get_context_chunks(document_id, max_chunks=10)
        if not chunks:
            raise NotelyError(
                "No indexed study material available to generate a quiz. Please upload or index a document first."
            )

        quiz_items = self.provider.generate_quiz(chunks, count=question_count)
        return {
            "document_id": document_id,
            "total_questions": len(quiz_items),
            "is_ai_generated": not self.provider.is_mock,
            "provider": self.provider.provider_name,
            "questions": quiz_items,
        }

    def generate_revision_notes(
        self,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate structured revision sheets including definitions, concepts, formulas, and comparisons."""
        chunks = self._get_context_chunks(document_id, max_chunks=15)
        if not chunks:
            raise NotelyError(
                "No indexed study material available to generate revision notes. Please upload or index a document first."
            )

        notes = self.provider.generate_revision_notes(chunks)
        return {
            "document_id": document_id,
            "is_ai_generated": not self.provider.is_mock,
            "provider": self.provider.provider_name,
            "definitions": notes.get("definitions", []),
            "key_concepts": notes.get("key_concepts", []),
            "formulas_or_rules": notes.get("formulas_or_rules", []),
            "common_comparisons": notes.get("common_comparisons", []),
            "potential_exam_questions": notes.get("potential_exam_questions", []),
        }
