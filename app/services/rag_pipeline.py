"""End-to-end Retrieval-Augmented Generation (RAG) pipeline."""

from typing import Any

from app.config import settings
from app.core.exceptions import NotelyError
from app.core.logging import get_logger
from app.schemas import Citation
from app.services.llm_provider import BaseLLMProvider, get_llm_provider
from app.services.retriever import SemanticRetriever

logger = get_logger(__name__)


class RAGPipeline:
    """Orchestrates query validation, semantic retrieval, evidence grounding, and cited answer generation."""

    def __init__(
        self,
        retriever: SemanticRetriever | None = None,
        provider: BaseLLMProvider | None = None,
    ) -> None:
        self.retriever = retriever or SemanticRetriever()
        self.provider = provider or get_llm_provider()

    def answer_question(
        self,
        query: str,
        document_id: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Execute complete RAG flow for a user query."""
        clean_query = query.strip()
        if not clean_query:
            raise NotelyError("Question cannot be empty.")

        k = top_k or settings.notely_default_top_k

        # 1. Retrieve evidence passages
        retrieved_chunks = self.retriever.retrieve(
            query=clean_query,
            top_k=k,
            document_id=document_id,
        )

        # 2. Check for empty knowledge base or zero results
        if not retrieved_chunks:
            return {
                "query": clean_query,
                "answer": (
                    "Based on the provided notes and documents, there is insufficient evidence to answer this question. "
                    "No relevant indexed passages were found matching your query."
                ),
                "citations": [],
                "is_ai_generated": not self.provider.is_mock,
                "provider": self.provider.provider_name,
                "sufficient_evidence": False,
            }

        # 3. Format citations from retrieved chunks
        citations: list[Citation] = []
        for chunk in retrieved_chunks:
            citations.append(
                Citation(
                    document_id=chunk.get("document_id", ""),
                    document_name=chunk.get("document_name", "Pasted Notes"),
                    page_number=chunk.get("page_number", 1),
                    chunk_id=chunk.get("chunk_id", ""),
                    excerpt=chunk.get("text", "")[:350] + ("..." if len(chunk.get("text", "")) > 350 else ""),
                    relevance_score=chunk.get("score"),
                )
            )

        # 4. Generate grounded answer
        answer, sufficient = self.provider.generate_answer(clean_query, retrieved_chunks)

        return {
            "query": clean_query,
            "answer": answer,
            "citations": [c.model_dump() for c in citations],
            "is_ai_generated": not self.provider.is_mock,
            "provider": self.provider.provider_name,
            "sufficient_evidence": sufficient,
        }
