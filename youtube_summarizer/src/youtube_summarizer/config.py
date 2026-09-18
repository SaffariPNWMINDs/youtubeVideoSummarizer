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

    # Google OAuth
    google_client_id: Optional[str] = None
    google_client_secret: Optional[str] = None
    google_redirect_uri: Optional[str] = None
    google_project_id: Optional[str] = None
    google_auth_uri: Optional[str] = None
    google_token_uri: Optional[str] = None
    google_auth_provider_x509_cert_url: Optional[str] = None

    # Pipeline tuning
    max_videos: int = 10
    # ~75k tokens. gpt-4o-mini (128k context) and claude-haiku-4-5 (200k
    # context) both have far more headroom than that — 50k was cutting off
    # real content on anything past ~30-45 min of video. This still leaves
    # a comfortable margin below either model's window for prompt + output.
    max_transcript_chars: int = 300_000
    pipeline_max_workers: int = 5        # concurrent workers for the map stage (transcript fetch + summarize)
    claude_per_video_model: str = "claude-haiku-4-5"
    claude_aggregate_model: str = "claude-sonnet-4-6"
    openai_per_video_model: str = "gpt-4o-mini"
    openai_aggregate_model: str = "gpt-4o"


# Singleton — import this everywhere instead of creating new instances
settings = Settings()
