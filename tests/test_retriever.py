"""Tests for vector store indexing, cosine retrieval, and document deletion."""

from app.models.chunk import DocumentChunk
from app.services.retriever import SemanticRetriever
from app.services.vector_store import VectorStoreService


def test_empty_vector_collection_retrieval(isolated_vector_store: VectorStoreService) -> None:
    retriever = SemanticRetriever(vector_store=isolated_vector_store)
    results = retriever.retrieve("Any query on empty database")
    assert results == []


def test_indexing_and_similarity_retrieval(isolated_vector_store: VectorStoreService) -> None:
    c1 = DocumentChunk(
        id="chunk_bio_1",
        document_id="doc_bio",
        document_name="Biology.pdf",
        chunk_index=0,
        page_number=1,
        text="Mitochondria are known as the powerhouse of eukaryotic cells and synthesize ATP.",
        char_count=80,
    )
    c2 = DocumentChunk(
        id="chunk_cs_1",
        document_id="doc_cs",
        document_name="Algorithms.pdf",
        chunk_index=0,
        page_number=1,
        text="Quicksort is a divide-and-conquer sorting algorithm with average O(n log n) runtime.",
        char_count=84,
    )

    isolated_vector_store.add_chunks([c1, c2])
    assert isolated_vector_store.get_chunk_count() == 2

    retriever = SemanticRetriever(vector_store=isolated_vector_store)

    # Query about biology should retrieve bio chunk first
    bio_results = retriever.retrieve("mitochondria ATP powerhouse", top_k=2)
    assert len(bio_results) == 2
    assert bio_results[0]["document_id"] == "doc_bio"
    assert bio_results[0]["score"] > 0.4

    # Query with document filter
    cs_filtered = retriever.retrieve("energy cell", top_k=2, document_id="doc_cs")
    assert len(cs_filtered) == 1
    assert cs_filtered[0]["document_id"] == "doc_cs"


def test_document_deletion_from_vector_store(isolated_vector_store: VectorStoreService) -> None:
    c1 = DocumentChunk(
        id="chunk_del_1",
        document_id="doc_del",
        document_name="Del.txt",
        chunk_index=0,
        page_number=1,
        text="Sample text to delete.",
        char_count=22,
    )
    isolated_vector_store.add_chunks([c1])
    assert isolated_vector_store.get_chunk_count() == 1

    # Delete
    deleted_count = isolated_vector_store.delete_document_chunks("doc_del")
    assert deleted_count == 1
    assert isolated_vector_store.get_chunk_count() == 0
