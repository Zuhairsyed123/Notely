"""Tests for health check endpoint."""

from fastapi.testclient import TestClient


def test_health_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert "embedding_model" in data
    assert "llm_provider" in data
    assert "document_count" in data
    assert "chunk_count" in data
