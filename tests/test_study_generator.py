"""Tests for study tools: flashcards, MCQ quiz, and revision notes."""

import pytest

from app.core.exceptions import NotelyError
from app.models.chunk import DocumentChunk
from app.services.llm_provider import MockLLMProvider
from app.services.study_generator import StudyGenerator
from app.services.vector_store import VectorStoreService


def test_flashcards_generation(isolated_vector_store: VectorStoreService) -> None:
    chunk = DocumentChunk(
        id="c1",
        document_id="doc1",
        document_name="Physics.pdf",
        chunk_index=0,
        page_number=2,
        text="Newton's second law is defined as force equals mass times acceleration (F=ma).",
        char_count=80,
    )
    isolated_vector_store.add_chunks([chunk])

    gen = StudyGenerator(vector_store=isolated_vector_store, provider=MockLLMProvider())
    res = gen.generate_flashcards(document_id="doc1", count=2)

    assert "flashcards" in res
    assert len(res["flashcards"]) > 0
    card = res["flashcards"][0]
    assert "question" in card
    assert "answer" in card
    assert "source_citation" in card
    assert "Physics.pdf" in card["source_citation"]


def test_quiz_generation(isolated_vector_store: VectorStoreService) -> None:
    chunk = DocumentChunk(
        id="c1",
        document_id="doc1",
        document_name="Genetics.pdf",
        chunk_index=0,
        page_number=5,
        text="Deoxyribonucleic acid carries genetic instructions for development and functioning.",
        char_count=84,
    )
    isolated_vector_store.add_chunks([chunk])

    gen = StudyGenerator(vector_store=isolated_vector_store, provider=MockLLMProvider())
    res = gen.generate_quiz(document_id="doc1", question_count=1)

    assert "questions" in res
    assert len(res["questions"]) == 1
    q = res["questions"][0]
    assert "question" in q
    assert len(q["options"]) == 4
    assert q["correct_answer"] in q["options"]
    assert "explanation" in q
    assert "source_citation" in q


def test_revision_notes_generation(isolated_vector_store: VectorStoreService) -> None:
    chunk = DocumentChunk(
        id="c1",
        document_id="doc1",
        document_name="Economics.pdf",
        chunk_index=0,
        page_number=1,
        text="Inflation refers to a general increase in prices and fall in the purchasing power of money. The Fisher equation is defined as real interest rate = nominal rate - inflation.",
        char_count=174,
    )
    isolated_vector_store.add_chunks([chunk])

    gen = StudyGenerator(vector_store=isolated_vector_store, provider=MockLLMProvider())
    res = gen.generate_revision_notes(document_id="doc1")

    assert "definitions" in res
    assert "key_concepts" in res
    assert "formulas_or_rules" in res
    assert len(res["definitions"]) > 0


def test_study_generator_empty_error(isolated_vector_store: VectorStoreService) -> None:
    gen = StudyGenerator(vector_store=isolated_vector_store, provider=MockLLMProvider())
    with pytest.raises(NotelyError):
        gen.generate_flashcards(document_id="nonexistent")
