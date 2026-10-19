"""Pydantic request and response schemas for Notely API endpoints."""

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health Schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(default="ok", json_schema_extra={"example": "ok"})
    version: str = Field(default="0.1.0")
    embedding_model: str
    llm_provider: str
    document_count: int
    chunk_count: int


# ---------------------------------------------------------------------------
# Document Schemas
# ---------------------------------------------------------------------------
class PasteNotesRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Title for pasted notes")
    text: str = Field(..., min_length=1, description="Raw text notes content")


class DocumentSummary(BaseModel):
    id: str
    name: str
    file_type: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    status: str
    created_at: str
    error_message: str | None = None


class DocumentListResponse(BaseModel):
    total_documents: int
    documents: list[DocumentSummary]


class DocumentDetailResponse(BaseModel):
    document: DocumentSummary
    full_text_preview: str
    pages: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Chunking & Explorer Schemas
# ---------------------------------------------------------------------------
class ChunkConfigRequest(BaseModel):
    chunk_size: int = Field(default=500, ge=50, le=4000, description="Chunk target character size")
    chunk_overlap: int = Field(default=100, ge=0, le=2000, description="Overlap in characters")


class ChunkItemResponse(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    page_number: int
    text: str
    char_count: int
    start_char: int
    end_char: int


class DocumentChunksResponse(BaseModel):
    document_id: str
    document_name: str
    chunk_size: int
    chunk_overlap: int
    total_chunks: int
    chunks: list[ChunkItemResponse]


# ---------------------------------------------------------------------------
# Summarization Schemas
# ---------------------------------------------------------------------------
SummaryModeType = Literal[
    "extractive",
    "ai_short",
    "ai_detailed",
    "ai_chapter",
    "ai_bullet_points",
    "ai_revision",
]


class SummarizeRequest(BaseModel):
    mode: SummaryModeType = Field(
        default="extractive",
        description="Summarization mode: extractive, ai_short, ai_detailed, ai_chapter, ai_bullet_points, ai_revision",
    )
    max_sentences: int = Field(
        default=5, ge=1, le=30, description="For extractive mode: number of sentences"
    )


class SummarizeResponse(BaseModel):
    document_id: str
    document_name: str
    mode: str
    summary: str
    is_ai_generated: bool
    provider: str
    char_count: int


# ---------------------------------------------------------------------------
# Search & Retrieval Schemas
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    document_id: str | None = Field(default=None, description="Optional document filter")
    top_k: int = Field(default=4, ge=1, le=20, description="Max results to return")


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    chunk_index: int
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[SearchResultItem]


# ---------------------------------------------------------------------------
# Chat / RAG Schemas
# ---------------------------------------------------------------------------
class Citation(BaseModel):
    document_id: str
    document_name: str
    page_number: int
    chunk_id: str
    excerpt: str
    relevance_score: float | None = None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question for the study material")
    document_id: str | None = Field(default=None, description="Optional document filter")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of evidence chunks to retrieve")


class ChatResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    is_ai_generated: bool
    provider: str
    sufficient_evidence: bool


# ---------------------------------------------------------------------------
# Study Tools Schemas
# ---------------------------------------------------------------------------
class FlashcardItem(BaseModel):
    question: str
    answer: str
    source_citation: str


class FlashcardsRequest(BaseModel):
    document_id: str | None = Field(default=None, description="Optional document filter")
    count: int = Field(default=5, ge=1, le=20, description="Number of flashcards to generate")


class FlashcardsResponse(BaseModel):
    document_id: str | None
    total_cards: int
    is_ai_generated: bool
    provider: str
    flashcards: list[FlashcardItem]


class MCQItem(BaseModel):
    id: int
    question: str
    options: list[str] = Field(..., min_length=2, max_length=5)
    correct_answer: str
    explanation: str
    source_citation: str


class QuizRequest(BaseModel):
    document_id: str | None = Field(default=None, description="Optional document filter")
    question_count: int = Field(default=5, ge=1, le=15, description="Number of quiz questions")


class QuizResponse(BaseModel):
    document_id: str | None
    total_questions: int
    is_ai_generated: bool
    provider: str
    questions: list[MCQItem]


class RevisionNotesRequest(BaseModel):
    document_id: str | None = Field(default=None, description="Optional document filter")


class RevisionDefinition(BaseModel):
    term: str
    definition: str
    source: str


class RevisionConcept(BaseModel):
    concept: str
    explanation: str
    source: str


class RevisionNotesResponse(BaseModel):
    document_id: str | None
    is_ai_generated: bool
    provider: str
    definitions: list[RevisionDefinition]
    key_concepts: list[RevisionConcept]
    formulas_or_rules: list[str]
    common_comparisons: list[dict[str, str]]
    potential_exam_questions: list[str]
