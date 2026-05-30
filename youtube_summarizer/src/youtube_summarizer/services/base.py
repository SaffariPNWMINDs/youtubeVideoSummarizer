"""
Abstract base classes for all services.

WHY THIS MATTERS (OOP lesson):
  These ABCs define the *interface contract* — what a service must be able to do,
  not how it does it. This is the Strategy Pattern.

  The SearchPipeline only depends on these abstractions, never on concrete classes.
  That means you can swap YouTubeSearchService for a MockSearchService in tests,
  or swap ClaudeSummarizerService for an OpenAISummarizerService — without touching
  any pipeline code. This is the Dependency Inversion Principle (the D in SOLID).
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from youtube_summarizer.models.video import Video
from youtube_summarizer.models.transcript import Transcript
from youtube_summarizer.models.summary import VideoSummary, AggregatedSummary


class BaseVideoSearchService(ABC):
    """Contract: given a query, return a ranked list of Videos."""

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[Video]:
        """
        Search for videos matching the query.
        Returns videos sorted by relevance/view count, best first.
        """
        ...


class BaseTranscriptService(ABC):
    """Contract: given a Video, return its Transcript (or None if unavailable)."""

    @abstractmethod
    def fetch(self, video: Video) -> Optional[Transcript]:
        """
        Fetch the transcript for a video.
        Returns None instead of raising if the transcript is genuinely unavailable.
        Only raises for unexpected infrastructure errors.
        """
        ...


class BaseSummarizerService(ABC):
    """Contract: summarize a single video, then aggregate N summaries."""

    @abstractmethod
    def summarize_video(self, video: Video, transcript: Transcript) -> VideoSummary:
        """Produce a structured summary for one video."""
        ...

    @abstractmethod
    def aggregate_summaries(
        self, query: str, summaries: List[VideoSummary]
    ) -> AggregatedSummary:
        """Synthesize all per-video summaries into one final report."""
        ...
