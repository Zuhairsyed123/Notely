"""Embedding service using Sentence Transformers with lazy loading and caching."""

from typing import Any

from app.config import settings
from app.core.exceptions import EmbeddingModelError
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Manages local embedding generation with lazy model initialization and vector normalization."""

    _instance: Any = None
    _model: Any = None

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.notely_embedding_model

    def _get_model(self) -> Any:
        """Lazily loads the SentenceTransformer model on first usage."""
        if EmbeddingService._model is None:
            logger.info("Initializing embedding model: %s", self.model_name)
            try:
                from sentence_transformers import SentenceTransformer

                EmbeddingService._model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.error("Failed to load embedding model '%s': %s", self.model_name, str(e))
                raise EmbeddingModelError(
                    f"Could not load embedding model '{self.model_name}'. "
                    f"Check internet connection or model cache. Error: {e}"
                ) from e
        return EmbeddingService._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized dense embeddings for a batch of text strings."""
        if not texts:
            return []

        model = self._get_model()
        try:
            # Normalize embeddings so dot product equals cosine similarity
            embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error("Error during batch embedding generation: %s", str(e))
            raise EmbeddingModelError(f"Embedding generation failed: {e}") from e

    def embed_query(self, query: str) -> list[float]:
        """Generate a normalized dense embedding for a single search or question query."""
        if not query or not query.strip():
            raise EmbeddingModelError("Query text cannot be empty.")
        embeddings = self.embed_texts([query.strip()])
        return embeddings[0]


# Singleton helper
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Retrieve the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
