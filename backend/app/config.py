from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    retrieval_min_similarity: float = Field(default=0.25, ge=0.0, le=1.0)
    database_url: str = (
        "postgresql+psycopg://researchshastra:researchshastra_dev"
        "@127.0.0.1:5432/researchshastra"
    )
    chroma_path: str = "./data/chroma"
    upload_dir: str = "./data/uploads"
    cors_origins: list[str] = ["http://localhost:3000"]
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        enable_decoding=False,
        extra="ignore",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
