"""
Configuration settings for TripPilot
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""

    # API Keys (optional - will use mock data if not provided)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # App settings
    app_name: str = "TripPilot"
    app_description: str = "AI-Powered Travel Planning Assistant"
    debug: bool = True

    # UI settings
    theme: str = "soft"
    max_chat_history: int = 50

    # Agent settings
    agent_model: str = "gpt-4"  # Default model
    agent_temperature: float = 0.7
    max_iterations: int = 10

    # Data paths
    data_dir: str = "data"
    cache_dir: str = ".cache"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
