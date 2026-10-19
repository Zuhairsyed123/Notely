"""Tests for local embedding service."""

import math

import pytest

from app.core.exceptions import EmbeddingModelError
from app.services.embeddings import EmbeddingService


def test_embedding_generation_and_normalization() -> None:
    svc = EmbeddingService()
    texts = ["Cellular biology and mitochondria", "Computer networks and routing"]
    vectors = svc.embed_texts(texts)

    assert len(vectors) == 2
    dim = len(vectors[0])
    assert dim > 100  # Typically 384 for MiniLM

    # Verify L2 normalization: norm should equal 1.0
    norm1 = math.sqrt(sum(x * x for x in vectors[0]))
    norm2 = math.sqrt(sum(x * x for x in vectors[1]))
    assert pytest.approx(norm1, rel=1e-3) == 1.0
    assert pytest.approx(norm2, rel=1e-3) == 1.0


def test_embed_empty_query() -> None:
    svc = EmbeddingService()
    with pytest.raises(EmbeddingModelError):
        svc.embed_query("")
