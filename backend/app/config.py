"""
Application configuration.
Loads settings from environment variables / .env file.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # App
    app_name: str = "Eye Hear U API"
    debug: bool = False

    # Firebase
    firebase_credentials_path: str = "firebase-credentials.json"
    firebase_project_id: str = ""

    # ML Model
    model_path: str = "../ml/checkpoints/best_model.pt"
    model_device: str = "cpu"  # "cpu" or "cuda" or "mps"

    # CORS - allowed origins for the mobile app
    cors_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
