"""
Confirms both summarizer services scale key_points to the video's length
instead of always asking for a fixed count, and size max_tokens to match.
"""

import json
from unittest.mock import MagicMock, patch

from youtube_summarizer.models.transcript import Transcript, TranscriptSegment
from youtube_summarizer.models.video import Video
from youtube_summarizer.services.openai_summarizer_service import OpenAISummarizerService
from youtube_summarizer.services.summarizer_service import ClaudeSummarizerService


def make_video() -> Video:
    return Video(
        video_id="v1",
        title="Test Video",
        channel_name="Test Channel",
        view_count=100_000,
        published_at="2024-01-01T00:00:00Z",
    )


def make_transcript(duration_seconds: float) -> Transcript:
    return Transcript(
        video_id="v1",
        segments=[TranscriptSegment(text="content " * 50, start=0.0, duration=duration_seconds)],
    )


class TestClaudeSummarizerKeyPointScaling:

    def _summarize_with_mocked_client(self, duration_seconds: float, key_points: list[str]):
        fake_message = MagicMock()
        fake_message.content = [MagicMock(text=json.dumps({
            "key_points": key_points,
            "raw_summary": "Fake summary.",
        }))]
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_message

        with patch("youtube_summarizer.services.summarizer_service.anthropic.Anthropic", return_value=fake_client):
            service = ClaudeSummarizerService()
            service.summarize_video(make_video(), make_transcript(duration_seconds))

        return fake_client.messages.create.call_args.kwargs

    def test_short_video_requests_fewer_key_points_and_smaller_budget(self):
        call_kwargs = self._summarize_with_mocked_client(duration_seconds=2 * 60, key_points=["a", "b", "c"])
        assert "exactly 3 items" in call_kwargs["messages"][0]["content"]
        assert call_kwargs["max_tokens"] == 600

    def test_long_video_requests_more_key_points_and_bigger_budget(self):
        # 19 min — long enough to scale up the key point count, but still
        # under the 20-min chunking threshold, so this stays a single call.
        call_kwargs = self._summarize_with_mocked_client(
            duration_seconds=19 * 60, key_points=[f"point {i}" for i in range(7)]
        )
        assert "exactly 7 items" in call_kwargs["messages"][0]["content"]
        assert call_kwargs["max_tokens"] > 600


class TestOpenAISummarizerKeyPointScaling:

    def _summarize_with_mocked_client(self, duration_seconds: float, key_points: list[dict]):
        fake_response = MagicMock()
        fake_response.choices = [MagicMock(message=MagicMock(content=json.dumps({
            "key_points": key_points,
            "raw_summary": "Fake summary.",
            "categories": [],
        })))]
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch("youtube_summarizer.services.openai_summarizer_service.OpenAI", return_value=fake_client):
            service = OpenAISummarizerService()
            service.summarize_video(make_video(), make_transcript(duration_seconds))

        return fake_client.chat.completions.create.call_args.kwargs

    def test_short_video_requests_fewer_key_points_and_smaller_budget(self):
        call_kwargs = self._summarize_with_mocked_client(
            duration_seconds=2 * 60,
            key_points=[
                {"text": "a", "timestamp_marker": "00:01"},
                {"text": "b", "timestamp_marker": "00:02"},
                {"text": "c", "timestamp_marker": "00:03"},
            ],
        )
        assert "exactly 3 items" in call_kwargs["messages"][0]["content"]
        assert call_kwargs["max_tokens"] == 800

    def test_long_video_requests_more_key_points_and_bigger_budget(self):
        # 19 min — long enough to scale up the key point count, but still
        # under the 20-min chunking threshold, so this stays a single call.
        call_kwargs = self._summarize_with_mocked_client(
            duration_seconds=19 * 60,
            key_points=[{"text": f"point {i}", "timestamp_marker": f"00:{i:02d}"} for i in range(7)],
        )
        assert "exactly 7 items" in call_kwargs["messages"][0]["content"]
        assert call_kwargs["max_tokens"] > 800
