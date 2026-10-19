"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient


def test_api_document_paste_and_workflow(api_client: TestClient) -> None:
    # 1. Paste notes
    payload = {
        "title": "Data Structures: Hash Tables",
        "text": (
            "A hash table is defined as a data structure that maps keys to values using a hash function. "
            "Collisions occur when two different keys generate the identical hash index. "
            "Chaining and open addressing are two primary collision resolution techniques."
        ),
    }
    paste_res = api_client.post("/documents/paste", json=payload)
    assert paste_res.status_code == 201
    doc_data = paste_res.json()
    doc_id = doc_data["id"]
    assert doc_data["name"] == "Data Structures: Hash Tables"
    assert doc_data["chunk_count"] > 0

    # 2. List documents
    list_res = api_client.get("/documents")
    assert list_res.status_code == 200
    docs = list_res.json()["documents"]
    assert any(d["id"] == doc_id for d in docs)

    # 3. Get document detail
    detail_res = api_client.get(f"/documents/{doc_id}")
    assert detail_res.status_code == 200
    assert len(detail_res.json()["pages"]) == 1

    # 4. Search
    search_res = api_client.post("/search", json={"query": "collision resolution chaining", "top_k": 2})
    assert search_res.status_code == 200
    assert search_res.json()["total_results"] > 0

    # 5. Chat (RAG)
    chat_res = api_client.post(
        "/chat", json={"query": "What are two collision resolution techniques?", "document_id": doc_id}
    )
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert chat_data["sufficient_evidence"] is True
    assert len(chat_data["citations"]) > 0

    # 6. Summarize
    sum_res = api_client.post(
        f"/documents/{doc_id}/summarize",
        json={"mode": "extractive", "max_sentences": 2},
    )
    assert sum_res.status_code == 200
    assert "Extractive Summary" in sum_res.json()["summary"]

    # 7. Chunk Explorer
    chunk_res = api_client.post(
        f"/documents/{doc_id}/chunks",
        json={"chunk_size": 200, "chunk_overlap": 50},
    )
    assert chunk_res.status_code == 200
    assert chunk_res.json()["total_chunks"] > 0

    # 8. Flashcards
    flash_res = api_client.post("/study/flashcards", json={"document_id": doc_id, "count": 2})
    assert flash_res.status_code == 200
    assert len(flash_res.json()["flashcards"]) > 0

    # 9. Quiz
    quiz_res = api_client.post("/study/quiz", json={"document_id": doc_id, "question_count": 1})
    assert quiz_res.status_code == 200
    assert len(quiz_res.json()["questions"]) == 1

    # 10. Delete Document
    del_res = api_client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200

    # Verify 404 after deletion
    detail_after = api_client.get(f"/documents/{doc_id}")
    assert detail_after.status_code == 404


def test_api_upload_validation_errors(api_client: TestClient) -> None:
    # Empty file
    res = api_client.post(
        "/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert res.status_code == 400

    # Unsupported format
    res2 = api_client.post(
        "/documents/upload",
        files={"file": ("notes.exe", b"fake binary", "application/octet-stream")},
    )
    assert res2.status_code == 400
