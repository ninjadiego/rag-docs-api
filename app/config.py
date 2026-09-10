from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM access goes through go-ai-gateway (or any OpenAI-compatible endpoint).
    llm_base_url: str = Field(default="http://localhost:8080/v1", alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="gw_live_change-me", alias="LLM_API_KEY")
    llm_model: str = Field(default="claude-sonnet-4-6", alias="LLM_MODEL")
    llm_max_tokens: int = Field(default=512, alias="LLM_MAX_TOKENS")

    # Vector store
    chroma_path: str = Field(default="./data/chroma", alias="CHROMA_PATH")
    collection_name: str = Field(default="documents", alias="COLLECTION_NAME")

    # Retrieval
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=4, alias="TOP_K")


@lru_cache
def get_settings() -> Settings:
    return Settings()
