from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    OLLAMA_URL: str = Field(default="http://localhost:11434")
    OPENAI_API_BASE: Optional[str] = Field(default=None)
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    GEMINI_API_KEY: Optional[str] = Field(default=None)
    MODEL_NAME: str = Field(default="gemini-2.5-flash")
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-large-en-v1.5")
    RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-base")
    FAISS_PATH: str = Field(default="./vector_store")
    BM25_INDEX_PATH: str = Field(default="./vector_store/bm25_index.pkl")
    PARENT_CHUNK_SIZE: int = Field(default=1500)
    CHILD_CHUNK_SIZE: int = Field(default=400)
    SIMILARITY_THRESHOLD: float = Field(default=0.4)
    LOG_LEVEL: str = Field(default="INFO")

settings = Settings()
