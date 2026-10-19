"""Tests for extractive and provider-backed document summarization."""

from app.models.document import Document
from app.services.llm_provider import MockLLMProvider
from app.services.summarizer import DocumentSummarizer


def test_extractive_summarization(sample_document: Document) -> None:
    summarizer = DocumentSummarizer(provider=MockLLMProvider())
    result = summarizer.summarize_document(sample_document, mode="extractive", max_sentences=2)

    assert result["mode"] == "extractive"
    assert result["is_ai_generated"] is False
    assert "Extractive Summary" in result["summary"]
    assert "Machine learning" in result["summary"] or "Supervised learning" in result["summary"]


def test_ai_summarization_modes(sample_document: Document) -> None:
    provider = MockLLMProvider()
    summarizer = DocumentSummarizer(provider=provider)

    modes = ["ai_short", "ai_detailed", "ai_bullet_points", "ai_revision", "ai_chapter"]
    for mode in modes:
        res = summarizer.summarize_document(sample_document, mode=mode)
        assert res["mode"] == mode
        assert res["char_count"] > 10
        assert res["summary"] is not None
