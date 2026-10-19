import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile
import uuid
from typing import Generator

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.document import Document, DocumentPage
from app.services.document_store import DocumentStore
from app.services.llm_provider import MockLLMProvider
from app.services.vector_store import VectorStoreService


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory wiped clean after test execution."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_path:
        yield Path(tmp_path)


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate in-memory PDF with extractable multi-page lecture notes."""
    doc = pymupdf.open()

    p1 = doc.new_page()
    p1.insert_text(
        (50, 72),
        "Photosynthesis Lecture Notes.\n"
        "Photosynthesis is defined as the biological process by which autotrophic organisms convert light energy into chemical energy stored in glucose.\n"
        "The light reactions take place in the thylakoid membranes of chloroplasts.",
    )

    p2 = doc.new_page()
    p2.insert_text(
        (50, 72),
        "Cellular Respiration and Energy Output.\n"
        "Cellular respiration refers to the metabolic pathway that breaks down glucose to produce adenosine triphosphate (ATP).\n"
        "The governing equation is C6H12O6 + 6O2 -> 6CO2 + 6H2O + ATP.\n"
        "In contrast to photosynthesis, respiration is a catabolic process.",
    )

    content = doc.tobytes()
    doc.close()
    return content


@pytest.fixture
def scanned_pdf_bytes() -> bytes:
    """Generate in-memory PDF containing no text layer (simulating a scanned image document)."""
    doc = pymupdf.open()
    p = doc.new_page()
    # Draw a rectangle without text
    p.draw_rect(pymupdf.Rect(50, 50, 200, 200), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    content = doc.tobytes()
    doc.close()
    return content


@pytest.fixture
def sample_txt_bytes() -> bytes:
    """Sample raw plain text bytes for testing."""
    text = (
        "Operating Systems: Virtual Memory Management.\n\n"
        "Paging is defined as a memory management scheme that eliminates the need for contiguous allocation of physical memory.\n"
        "A page table is used to map virtual addresses to physical frame addresses.\n\n"
        "Thrashing occurs when a computer spends more time paging than executing instructions."
    )
    return text.encode("utf-8")


@pytest.fixture
def isolated_doc_store(temp_dir: Path) -> DocumentStore:
    """Provide an isolated DocumentStore using a temporary metadata file."""
    meta_path = temp_dir / "metadata.json"
    return DocumentStore(metadata_path=meta_path)


@pytest.fixture
def isolated_vector_store() -> VectorStoreService:
    """Provide an isolated in-memory ChromaDB VectorStore with a unique collection per test."""
    col_name = f"test_col_{uuid.uuid4().hex[:12]}"
    return VectorStoreService(in_memory=True, collection_name=col_name)


@pytest.fixture
def mock_provider() -> MockLLMProvider:
    """Provide deterministic mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def sample_document() -> Document:
    """Provide a pre-constructed Document model instance."""
    p1 = DocumentPage(
        page_number=1,
        raw_text="Machine learning is defined as a branch of artificial intelligence.",
        cleaned_text="Machine learning is defined as a branch of artificial intelligence.",
        char_count=67,
    )
    p2 = DocumentPage(
        page_number=2,
        raw_text="Supervised learning involves training a model on labeled data.",
        cleaned_text="Supervised learning involves training a model on labeled data.",
        char_count=62,
    )
    return Document(
        id="test_doc_123",
        name="Intro to Machine Learning.pdf",
        file_type="pdf",
        content_hash="abc123hash",
        file_size_bytes=1024,
        page_count=2,
        chunk_count=2,
        status="indexed",
        pages=[p1, p2],
    )


@pytest.fixture
def api_client() -> TestClient:
    """FastAPI TestClient for endpoint integration tests."""
    return TestClient(app)
