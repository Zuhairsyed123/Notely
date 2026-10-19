"""Application configuration settings for Notely."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory for the repository
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    api_base_url: str = "http://127.0.0.1:8000"

    # Storage paths
    notely_data_dir: Path = BASE_DIR / "data"
    notely_chroma_dir: Path = BASE_DIR / "data" / "chroma"
    documents_dir: Path = BASE_DIR / "data" / "documents"
    metadata_file: Path = BASE_DIR / "data" / "metadata.json"

    # Embedding and Vector store
    notely_embedding_model: str = "all-MiniLM-L6-v2"
    notely_default_chunk_size: int = 500
    notely_default_chunk_overlap: int = 100
    notely_default_top_k: int = 4

    # LLM Provider settings
    notely_llm_provider: Literal["mock", "openai", "gemini", "groq"] = "mock"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"

    # Limits and constraints
    max_upload_size_bytes: int = 25 * 1024 * 1024  # 25 MB

    def ensure_directories(self) -> None:
        """Create necessary persistent storage folders if they do not exist."""
        self.notely_data_dir.mkdir(parents=True, exist_ok=True)
        self.notely_chroma_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
