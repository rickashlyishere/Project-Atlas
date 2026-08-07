from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration."""

    APP_NAME: str = "Project Atlas"
    APP_VERSION: str = "0.1.0"

    DEBUG: bool = True

    DATA_DIR: Path = Path("data")
    CACHE_DIR: Path = Path("cache")
    LOG_DIR: Path = Path("logs")
    MODEL_DIR: Path = Path("models")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()