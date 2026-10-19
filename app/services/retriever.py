"""Semantic retriever for fetching contextually relevant chunks from the vector store."""

from typing import Any

from app.core.logging import get_logger
from app.services.vector_store import VectorStoreService, get_vector_store

logger = get_logger(__name__)


class SemanticRetriever:
    """Performs semantic similarity retrieval backed by ChromaDB vector store."""

    def __init__(self, vector_store: VectorStoreService | None = None) -> None:
        self.vector_store = vector_store or get_vector_store()

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k most semantically relevant chunks for a given query."""
        clean_query = query.strip()
        if not clean_query:
            return []

        logger.info(
            "Executing semantic retrieval for query (len=%d), top_k=%d, doc_filter=%s",
            len(clean_query),
            top_k,
            document_id[:8] if document_id else "None",
        )

        results = self.vector_store.query_similarity(
            query=clean_query,
            top_k=top_k,
            document_id=document_id,
        )

        logger.info("Retrieved %d relevant chunks", len(results))
        return results
