"""
Unit tests for YouTubeTranscriptService.

Covers the language fallback: when a video has no English caption track,
the service should fall back to whatever track IS available (e.g. Persian)
instead of skipping the video.
"""

from unittest.mock import MagicMock, patch

from youtube_transcript_api import (
    FetchedTranscript,
    FetchedTranscriptSnippet,
    NoTranscriptFound,
    TranscriptsDisabled,
)

from youtube_summarizer.models.video import Video
from youtube_summarizer.services.transcript_service import YouTubeTranscriptService


def make_video(video_id: str = "v1") -> Video:
    return Video(
        video_id=video_id,
        title="Test Video",
        channel_name="Test Channel",
        view_count=100_000,
        published_at="2024-01-01T00:00:00Z",
    )


def make_fetched(video_id: str, language_code: str, language: str, text: str) -> FetchedTranscript:
    return FetchedTranscript(
        snippets=[FetchedTranscriptSnippet(text=text, start=0.0, duration=2.0)],
        video_id=video_id,
        language=language,
        language_code=language_code,
        is_generated=False,
    )


class TestYouTubeTranscriptService:

    def test_fetches_english_directly_when_available(self):
        fake_api = MagicMock()
        fake_api.fetch.return_value = make_fetched("v1", "en", "English", "hello world")

        with patch(
            "youtube_summarizer.services.transcript_service.YouTubeTranscriptApi",
            return_value=fake_api,
        ):
            result = YouTubeTranscriptService().fetch(make_video())

        assert result is not None
        assert result.language == "en"
        assert result.full_text == "hello world"
        fake_api.list.assert_not_called()

    def test_falls_back_to_non_english_track_when_no_english_available(self):
        fake_api = MagicMock()
        fake_api.fetch.side_effect = NoTranscriptFound("v1", ["en"], MagicMock())

        farsi_transcript = MagicMock()
        farsi_transcript.language_code = "fa"
        farsi_transcript.fetch.return_value = make_fetched("v1", "fa", "Persian", "سلام دنیا")
        fake_api.list.return_value = [farsi_transcript]

        with patch(
            "youtube_summarizer.services.transcript_service.YouTubeTranscriptApi",
            return_value=fake_api,
        ):
            result = YouTubeTranscriptService().fetch(make_video())

        assert result is not None
        assert result.language == "fa"
        assert "سلام" in result.full_text

    def test_returns_none_when_no_transcript_in_any_language(self):
        fake_api = MagicMock()
        fake_api.fetch.side_effect = NoTranscriptFound("v1", ["en"], MagicMock())
        fake_api.list.return_value = []   # nothing available in any language

        with patch(
            "youtube_summarizer.services.transcript_service.YouTubeTranscriptApi",
            return_value=fake_api,
        ):
            result = YouTubeTranscriptService().fetch(make_video())

        assert result is None

    def test_returns_none_when_transcripts_disabled(self):
        fake_api = MagicMock()
        fake_api.fetch.side_effect = TranscriptsDisabled("v1")

        with patch(
            "youtube_summarizer.services.transcript_service.YouTubeTranscriptApi",
            return_value=fake_api,
        ):
            result = YouTubeTranscriptService().fetch(make_video())

        assert result is None
