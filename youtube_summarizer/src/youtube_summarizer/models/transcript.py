from pydantic import BaseModel, computed_field
from typing import List


class TranscriptSegment(BaseModel):
    """A single timed segment from YouTube's caption data."""

    text: str
    start: float        # seconds from video start
    duration: float     # length of this segment in seconds

    @property
    def end(self) -> float:
        return self.start + self.duration


class Transcript(BaseModel):
    """
    Full transcript for one video, composed of timed segments.
    Provides helper properties for downstream processing.
    """

    video_id: str
    segments: List[TranscriptSegment]
    language: str = "en"

    @computed_field
    @property
    def full_text(self) -> str:
        """Plain text — all segments joined. Used for LLM input."""
        return " ".join(seg.text.strip() for seg in self.segments if seg.text.strip())

    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.full_text)

    @computed_field
    @property
    def word_count(self) -> int:
        return len(self.full_text.split())

    def truncate(self, max_chars: int) -> str:
        """
        Returns the full text truncated to max_chars.
        Tries to cut at a sentence boundary to avoid mid-sentence truncation.
        """
        text = self.full_text
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.8:       # only cut at sentence if close enough
            return truncated[: last_period + 1]
        return truncated
