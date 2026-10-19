"""Tests for end-to-end RAG question answering pipeline."""

from app.models.chunk import DocumentChunk
from app.services.llm_provider import MockLLMProvider
from app.services.rag_pipeline import RAGPipeline
from app.services.retriever import SemanticRetriever
from app.services.vector_store import VectorStoreService


def test_rag_pipeline_with_grounded_citations(isolated_vector_store: VectorStoreService) -> None:
    chunk = DocumentChunk(
        id="chunk_chem_1",
        document_id="doc_chem",
        document_name="Chemistry.pdf",
        chunk_index=0,
        page_number=4,
        text="Avogadro's constant is defined as 6.022 x 10^23 particles per mole of substance.",
        char_count=82,
    )
    isolated_vector_store.add_chunks([chunk])

    retriever = SemanticRetriever(vector_store=isolated_vector_store)
    provider = MockLLMProvider()
    pipeline = RAGPipeline(retriever=retriever, provider=provider)

    res = pipeline.answer_question(query="What is Avogadro's constant?")

    assert res["sufficient_evidence"] is True
    assert len(res["citations"]) == 1
    citation = res["citations"][0]
    assert citation["document_name"] == "Chemistry.pdf"
    assert citation["page_number"] == 4
    assert citation["chunk_id"] == "chunk_chem_1"
    assert "6.022" in citation["excerpt"]
    assert "Chemistry.pdf" in res["answer"]


def test_rag_insufficient_evidence(isolated_vector_store: VectorStoreService) -> None:
    # Empty store
    retriever = SemanticRetriever(vector_store=isolated_vector_store)
    provider = MockLLMProvider()
    pipeline = RAGPipeline(retriever=retriever, provider=provider)

    res = pipeline.answer_question(query="What is the speed of light in vacuum?")
    assert res["sufficient_evidence"] is False
    assert "insufficient evidence" in res["answer"].lower()
    assert res["citations"] == []
