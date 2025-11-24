"""
Configuration management for TripPilot.

Uses Pydantic Settings for type-safe configuration with environment variable support.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "TripPilot"
    app_version: str = "2.0.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # LLM Configuration
    llm_provider: Literal["openai", "anthropic", "litellm"] = "openai"
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    default_model: str = "gpt-4o"
    fast_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096

    # Vector Database
    vector_db: Literal["chromadb", "lancedb"] = "chromadb"
    chromadb_path: str = "./data/chromadb"
    lancedb_path: str = "./data/lancedb"
    embedding_model: str = "all-MiniLM-L6-v2"
    collection_name: str = "travel_knowledge"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    cors_origins: list[str] = ["*"]

    # Search Configuration
    max_search_results: int = 10
    search_timeout: int = 30

    # Agent Configuration
    max_agent_iterations: int = 10
    agent_timeout: int = 120

    # Observability
    enable_tracing: bool = True
    wandb_project: str = "trippilot"
    wandb_enabled: bool = False
    log_level: str = "INFO"

    # Data Sources
    tripadvisor_dataset: str = "argilla/tripadvisor-hotel-reviews"
    processed_data_path: str = "./data/processed"

    @property
    def llm_api_key(self) -> SecretStr | None:
        """Get the API key for the configured LLM provider."""
        if self.llm_provider == "openai":
            return self.openai_api_key
        elif self.llm_provider == "anthropic":
            return self.anthropic_api_key
        return None


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
