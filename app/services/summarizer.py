"""Document summarization service providing extractive summarization and provider-backed AI summarization."""

import re
from collections import Counter
from typing import Any

from app.core.logging import get_logger
from app.models.document import Document
from app.services.llm_provider import BaseLLMProvider, get_llm_provider

logger = get_logger(__name__)

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
}


class DocumentSummarizer:
    """Handles extractive sentence-scoring summarization and chunk-reducing LLM summarization."""

    def __init__(self, provider: BaseLLMProvider | None = None) -> None:
        self.provider = provider or get_llm_provider()

    @staticmethod
    def extractive_summary(text: str, max_sentences: int = 5) -> str:
        """Local, non-AI algorithm that scores and selects top salient sentences based on frequency."""
        if not text or not text.strip():
            return "No content available to summarize."

        # Extract words for frequency table
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        meaningful_words = [w for w in words if w not in STOPWORDS]

        if not meaningful_words:
            return text[:300] + "..." if len(text) > 300 else text

        word_freq = Counter(meaningful_words)
        max_freq = max(word_freq.values())
        norm_freq = {w: count / max_freq for w, count in word_freq.items()}

        # Split text into candidate sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
        if not sentences:
            return text[:400]

        scored_sentences: list[tuple[int, float, str]] = []
        for idx, sentence in enumerate(sentences):
            s_words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())
            if not s_words:
                continue

            # Compute TF score
            score = sum(norm_freq.get(w, 0.0) for w in s_words) / (len(s_words) ** 0.5)

            # Boost opening sentences of sections / paragraphs
            if idx == 0:
                score *= 1.3
            elif idx < 3:
                score *= 1.15

            scored_sentences.append((idx, score, sentence))

        # Select top N sentences sorted by original chronological position
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = scored_sentences[:max_sentences]
        top_sentences.sort(key=lambda x: x[0])

        summary_text = " ".join([item[2] for item in top_sentences])
        return (
            "**Extractive Summary (Frequency-Based Salience):**\n\n"
            + summary_text
            + f"\n\n*(Selected {len(top_sentences)} representative sentences from document text without external LLM processing.)*"
        )

    def summarize_document(
        self,
        document: Document,
        mode: str = "extractive",
        max_sentences: int = 5,
    ) -> dict[str, Any]:
        """Generate summary for a full document, handling chunk reduction for large documents."""
        full_text = document.get_full_text()
        if not full_text.strip():
            return {
                "document_id": document.id,
                "document_name": document.name,
                "mode": mode,
                "summary": "Document contains no text to summarize.",
                "is_ai_generated": False,
                "provider": "none",
                "char_count": 0,
            }

        # Extractive mode
        if mode == "extractive":
            summary = self.extractive_summary(full_text, max_sentences=max_sentences)
            return {
                "document_id": document.id,
                "document_name": document.name,
                "mode": mode,
                "summary": summary,
                "is_ai_generated": False,
                "provider": "Extractive (Local TF-Salience)",
                "char_count": len(summary),
            }

        # AI-based summarization (via active provider)
        logger.info("Generating AI summary with mode='%s', provider='%s'", mode, self.provider.provider_name)

        # Handle large documents using hierarchical map-reduce if length > 8000 chars
        if len(full_text) > 8000:
            logger.info("Document exceeds 8,000 characters; running chunk-reduction summarization")
            # Chunk reduction
            section_chunks = [
                page.cleaned_text for page in document.pages if len(page.cleaned_text.strip()) > 100
            ]
            intermediate_summaries = []
            for sc in section_chunks[:8]:  # Limit to first 8 pages for rate/context safety
                intermediate_summaries.append(self.provider.generate_summary(sc[:2000], "ai_short"))

            combined_intermediate = "\n\n".join(intermediate_summaries)
            final_summary = self.provider.generate_summary(combined_intermediate, mode)
        else:
            final_summary = self.provider.generate_summary(full_text, mode)

        return {
            "document_id": document.id,
            "document_name": document.name,
            "mode": mode,
            "summary": final_summary,
            "is_ai_generated": not self.provider.is_mock,
            "provider": self.provider.provider_name,
            "char_count": len(final_summary),
        }
