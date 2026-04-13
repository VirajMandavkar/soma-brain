"""
SOMA Brain — Central Configuration

All settings loaded from environment variables via .env file.
Pydantic validates types at startup — if a required variable is missing
or has the wrong type, the app fails fast with a clear error message
instead of crashing midway through a simulation.
"""

from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings — every field maps to an environment variable."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- API Keys ---
    ncbi_api_key: str = Field(default="", description="PubMed E-utilities API key")
    semantic_scholar_api_key: str = Field(default="", description="Semantic Scholar API key")
    huggingface_token: str = Field(default="", description="HuggingFace access token")

    # --- Neo4j ---
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="soma_brain_neo4j_2026")

    # --- PostgreSQL ---
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5433)
    postgres_db: str = Field(default="soma_brain")
    postgres_user: str = Field(default="soma")
    postgres_password: str = Field(default="soma_brain_pg_2026")

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_url_sync(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # --- Weaviate ---
    weaviate_url: str = Field(default="http://localhost:8080")

    # --- Redis ---
    redis_url: str = Field(default="redis://localhost:6379/0")

    # --- AI Model ---
    ollama_base_url: str = Field(default="http://localhost:11434")
    gemma_model: str = Field(default="gemma3:4b")

    # --- Modal ---
    modal_token_id: str = Field(default="")
    modal_token_secret: str = Field(default="")

    # --- Application ---
    secret_key: str = Field(default="change-me-in-production-use-32-chars-minimum")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiry_hours: int = Field(default=8)
    log_level: str = Field(default="INFO")

    # --- Monte Carlo defaults ---
    mc_patient_samples: int = Field(default=100, description="N: patient parameter samples")
    mc_noise_seeds: int = Field(default=10, description="M: neural noise seeds per sample")
    mc_pk_samples: int = Field(default=20, description="K: pharmacokinetic samples per compound")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings instance. Cached so .env is only read once."""
    return Settings()
