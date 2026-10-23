import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Storage settings
    DATA_DIR: Path = BASE_DIR / "data"
    STORAGE_DIR: Path = BASE_DIR / "storage"
    
    # Models
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    
    # LLM Settings (OpenAI / OpenCode / Gemini compatible)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", os.getenv("GEMINI_API_KEY", "")))
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-3.5-flash" if (os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY", "").startswith("AIzaSy")) else "gpt-4o-mini")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini" if (os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY", "").startswith("AIzaSy")) else "openai")
    
    # Langfuse Observability
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    # Retrieval Tuning
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50
    HYBRID_ALPHA: float = 0.5  # Weight between dense (alpha) and lexical (1 - alpha)
    CANDIDATE_DEPTH: int = 25
    FINAL_CONTEXT_K: int = 10
    MAX_GAP_FILLING_ROUNDS: int = 2
    
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
