from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Central config loaded from .env file.
    Pydantic-settings validates types automatically — if YOUTUBE_API_KEY
    is missing, the app fails fast at startup with a clear error.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # API keys
    youtube_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Pipeline tuning
    max_videos: int = 10
    max_transcript_chars: int = 50_000   # ~12k tokens, safe for any model
    claude_per_video_model: str = "claude-haiku-4-5"
    claude_aggregate_model: str = "claude-sonnet-4-6"
    openai_per_video_model: str = "gpt-4o-mini"
    openai_aggregate_model: str = "gpt-4o"


# Singleton — import this everywhere instead of creating new instances
settings = Settings()
