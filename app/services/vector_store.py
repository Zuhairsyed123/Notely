"""Vector database integration using ChromaDB with persistent storage and cosine similarity."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.core.logging import get_logger
from app.models.chunk import DocumentChunk
from app.services.embeddings import EmbeddingService, get_embedding_service

logger = get_logger(__name__)


class VectorStoreService:
    """Manages persistent ChromaDB vector collections, chunk indexing, queries, and deletions."""

    COLLECTION_NAME = "notely_knowledge_base"

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        embedding_service: EmbeddingService | None = None,
        in_memory: bool = False,
        collection_name: str | None = None,
    ) -> None:
        self.in_memory = in_memory
        self.collection_name = collection_name or self.COLLECTION_NAME
        self.persist_dir = Path(persist_dir) if persist_dir else settings.notely_chroma_dir
        self.embedding_service = embedding_service or get_embedding_service()

        if self.in_memory:
            self._client = chromadb.EphemeralClient(
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        else:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        # Using cosine similarity metric
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore initialized (in_memory=%s, collection=%s) with %d existing vectors",
            self.in_memory,
            self.collection_name,
            self._collection.count(),
        )

    def clear(self) -> None:
        """Purge and recreate the current collection."""
        try:
            self._client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embed and upsert document chunks into the vector store."""
        if not chunks:
            return 0

        ids = [chunk.id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_metadata_dict() for chunk in chunks]

        # Generate embeddings
        embeddings = self.embedding_service.embed_texts(texts)

        # Upsert into Chroma (prevents duplicates by ID)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Indexed %d chunks into vector store", len(chunks))
        return len(chunks)

    def query_similarity(
        self,
        query: str,
        top_k: int = 4,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the collection using cosine similarity of query embedding against indexed chunks."""
        count = self._collection.count()
        if count == 0:
            logger.info("Vector store is empty, returning 0 results.")
            return []

        query_vector = self.embedding_service.embed_query(query)
        effective_k = min(top_k, count)

        where_filter = {"document_id": document_id} if document_id else None

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=effective_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results: list[dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return formatted_results

        matched_ids = results["ids"][0]
        matched_docs = results["documents"][0] if results.get("documents") else []
        matched_metas = results["metadatas"][0] if results.get("metadatas") else []
        matched_distances = results["distances"][0] if results.get("distances") else []

        for idx in range(len(matched_ids)):
            distance = matched_distances[idx] if idx < len(matched_distances) else 0.0
            # Cosine distance to similarity: similarity = 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - distance))

            meta = matched_metas[idx] if idx < len(matched_metas) else {}
            doc_text = matched_docs[idx] if idx < len(matched_docs) else ""

            formatted_results.append(
                {
                    "chunk_id": matched_ids[idx],
                    "document_id": meta.get("document_id", ""),
                    "document_name": meta.get("document_name", "Unknown Document"),
                    "page_number": int(meta.get("page_number", 1)),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                    "char_count": int(meta.get("char_count", len(doc_text))),
                    "text": doc_text,
                    "score": round(similarity, 4),
                    "distance": round(distance, 4),
                }
            )

        return formatted_results

    def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunk vectors belonging to a specific document ID."""
        try:
            # Query existing chunk count for this document
            existing = self._collection.get(where={"document_id": document_id})
            num_chunks = len(existing["ids"]) if existing and "ids" in existing else 0
            if num_chunks > 0:
                self._collection.delete(where={"document_id": document_id})
                logger.info("Deleted %d chunk vectors for document %s", num_chunks, document_id)
            return num_chunks
        except Exception as e:
            logger.error("Error deleting chunks for document %s: %s", document_id, str(e))
            return 0

    def get_chunk_count(self) -> int:
        """Return total number of vectors in the collection."""
        try:
            return self._collection.count()
        except Exception:
            return 0

    def get_chunks_for_document(self, document_id: str) -> list[dict[str, Any]]:
        """Retrieve stored chunks for a specific document without semantic search."""
        try:
            res = self._collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"],
            )
            items = []
            if res and "ids" in res:
                for idx, cid in enumerate(res["ids"]):
                    meta = res["metadatas"][idx] if res["metadatas"] else {}
                    text = res["documents"][idx] if res["documents"] else ""
                    items.append(
                        {
                            "chunk_id": cid,
                            "document_id": meta.get("document_id", document_id),
                            "document_name": meta.get("document_name", ""),
                            "page_number": int(meta.get("page_number", 1)),
                            "chunk_index": int(meta.get("chunk_index", idx)),
                            "char_count": int(meta.get("char_count", len(text))),
                            "text": text,
                            "start_char": int(meta.get("start_char", 0)),
                            "end_char": int(meta.get("end_char", 0)),
                        }
                    )
            # Sort by chunk_index
            items.sort(key=lambda x: x["chunk_index"])
            return items
        except Exception as e:
            logger.error("Error fetching chunks for document %s: %s", document_id, str(e))
            return []


# Singleton helper
_vector_store: VectorStoreService | None = None


def get_vector_store() -> VectorStoreService:
    """Retrieve the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store
