"""
Integration test for SearchPipeline.

This tests the pipeline's orchestration logic — skipping videos,
abort thresholds, correct sequencing — without hitting any real API.
All services are replaced with simple in-memory fakes.
"""

from typing import List, Optional

from youtube_summarizer.pipeline.search_pipeline import SearchPipeline
from youtube_summarizer.models.video import Video
from youtube_summarizer.models.transcript import Transcript, TranscriptSegment
from youtube_summarizer.models.summary import VideoSummary, AggregatedSummary
from youtube_summarizer.services.base import (
    BaseVideoSearchService,
    BaseTranscriptService,
    BaseSummarizerService,
)


# ── In-memory fakes (simpler than MagicMock for integration tests) ──────────

class FakeSearchService(BaseVideoSearchService):
    def __init__(self, videos: List[Video]):
        self._videos = videos

    def search(self, query: str, max_results: int = 10) -> List[Video]:
        return self._videos[:max_results]


class FakeTranscriptService(BaseTranscriptService):
    """Returns a transcript for every video except those in the skip set."""

    def __init__(self, skip_video_ids: Optional[set] = None):
        self._skip = skip_video_ids or set()

    def fetch(self, video: Video) -> Optional[Transcript]:
        if video.video_id in self._skip:
            return None
        return Transcript(
            video_id=video.video_id,
            segments=[TranscriptSegment(text="Fake transcript content.", start=0.0, duration=5.0)],
        )


class FakeSummarizerService(BaseSummarizerService):
    def summarize_video(self, video: Video, transcript: Transcript) -> VideoSummary:
        return VideoSummary(
            video_id=video.video_id,
            title=video.title,
            channel_name=video.channel_name,
            view_count=video.view_count,
            video_url=video.url,
            key_points=["Point A", "Point B", "Point C"],
            raw_summary="Fake summary.",
        )

    def aggregate_summaries(self, query: str, summaries: List[VideoSummary]) -> AggregatedSummary:
        return AggregatedSummary(
            query=query,
            video_summaries=summaries,
            final_summary="Aggregated fake summary.",
            key_takeaways=["Takeaway 1", "Takeaway 2", "Takeaway 3"],
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_video(video_id: str, title: str = "Test Video", views: int = 100_000) -> Video:
    return Video(
        video_id=video_id,
        title=title,
        channel_name="Test Channel",
        view_count=views,
        published_at="2024-01-01T00:00:00Z",
    )


def make_pipeline(videos, skip_ids=None, min_summaries=3) -> SearchPipeline:
    return SearchPipeline(
        search_service=FakeSearchService(videos),
        transcript_service=FakeTranscriptService(skip_video_ids=skip_ids),
        summarizer_service=FakeSummarizerService(),
        min_summaries=min_summaries,
    )


# ── Tests ────────────────────────────────────────────────────────────────────

class TestSearchPipeline:

    def test_happy_path_returns_summary(self):
        videos = [make_video(f"v{i}") for i in range(5)]
        pipeline = make_pipeline(videos, min_summaries=3)

        result = pipeline.run("machine learning")

        assert result is not None
        assert result.query == "machine learning"
        assert result.video_count == 5

    def test_skips_videos_without_transcripts(self):
        videos = [make_video("v1"), make_video("v2"), make_video("v3"),
                  make_video("v4"), make_video("v5")]
        # v1 and v2 have no transcripts
        pipeline = make_pipeline(videos, skip_ids={"v1", "v2"}, min_summaries=3)

        result = pipeline.run("test query")

        assert result is not None
        assert result.video_count == 3     # only 3 transcripts available

    def test_returns_none_when_too_few_summaries(self):
        videos = [make_video("v1"), make_video("v2")]
        # Only 2 videos, but min_summaries=3
        pipeline = make_pipeline(videos, min_summaries=3)

        result = pipeline.run("test query")

        assert result is None

    def test_returns_none_when_no_videos_found(self):
        pipeline = make_pipeline(videos=[], min_summaries=1)

        result = pipeline.run("obscure topic")

        assert result is None

    def test_result_contains_markdown(self):
        videos = [make_video(f"v{i}") for i in range(4)]
        pipeline = make_pipeline(videos, min_summaries=3)

        result = pipeline.run("python tutorial")
        markdown = result.to_markdown()

        assert "python tutorial" in markdown.lower()
        assert "# Summary" in markdown
        assert "## Key Takeaways" in markdown
