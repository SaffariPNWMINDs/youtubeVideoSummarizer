from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central config loaded from .env file.
    Pydantic-settings validates types automatically — if YOUTUBE_API_KEY
    is missing, the app fails fast at startup with a clear error.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # API keys
    youtube_api_key: str
    anthropic_api_key: str

    # Pipeline tuning
    max_videos: int = 10
    max_transcript_chars: int = 50_000   # ~12k tokens, safe for any model
    per_video_model: str = "claude-haiku-4-5"      # fast + cheap for per-video
    aggregate_model: str = "claude-sonnet-4-6"     # more capable for final synthesis


# Singleton — import this everywhere instead of creating new instances
settings = Settings()
