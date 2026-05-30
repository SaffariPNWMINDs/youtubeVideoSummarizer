from pydantic import BaseModel, computed_field
from typing import Optional


class Video(BaseModel):
    """
    Represents a single YouTube video returned by the search API.
    Immutable by default (Pydantic models are immutable unless you set model_config).
    """

    video_id: str
    title: str
    channel_name: str
    view_count: int
    published_at: str           # ISO 8601 string from YouTube API
    description: Optional[str] = None

    @computed_field                 # Pydantic v2 — derived property stored on the model
    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    def timestamp_url(self, seconds: float) -> str:
        """Deep link to a specific timestamp in the video."""
        return f"{self.url}&t={int(seconds)}s"

    def __str__(self) -> str:
        return f"[{self.view_count:,} views] {self.title} — {self.channel_name}"
