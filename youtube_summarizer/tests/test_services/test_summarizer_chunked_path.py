"""
Confirms both summarizer services actually chunk long videos: multiple
per-chunk LLM calls plus one synthesis call, instead of one call asked to
summarize an entire 60+ minute transcript at once. That single-call
approach is what produced key points clustered only in the first ~25
minutes of a 68-minute video even with the full transcript in context —
splitting into smaller, independently-summarized chunks is the fix.
"""

import json
import re
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


def make_long_transcript(total_minutes: float = 68.0, segment_seconds: float = 10.0) -> Transcript:
    """Many small segments spanning `total_minutes`, so chunk_transcript actually splits it."""
    num_segments = int(total_minutes * 60 / segment_seconds)
    return Transcript(
        video_id="v1",
        segments=[
            TranscriptSegment(text=f"content {i} " * 10, start=i * segment_seconds, duration=segment_seconds)
            for i in range(num_segments)
        ],
    )


def first_marker_in(prompt_content: str) -> str:
    """A marker that genuinely appears in this chunk's own transcript text —
    mirrors what a well-behaved model would copy, for realistic mocks."""
    return re.search(r"\[(\d{1,3}:\d{2})\]", prompt_content).group(1)


class TestClaudeChunkedSummarization:

    def test_long_video_triggers_multiple_calls_and_merges_key_points(self):
        call_count = {"n": 0}

        def fake_create(**kwargs):
            call_count["n"] += 1
            message = MagicMock()
            if "merging summaries" in kwargs["messages"][0]["content"]:
                message.content = [MagicMock(text=json.dumps({"raw_summary": "Merged overview."}))]
            else:
                message.content = [MagicMock(text=json.dumps({
                    "key_points": [f"point from call {call_count['n']}"],
                    "raw_summary": f"Chunk {call_count['n']} summary.",
                }))]
            return message

        fake_client = MagicMock()
        fake_client.messages.create.side_effect = fake_create

        with patch("youtube_summarizer.services.summarizer_service.anthropic.Anthropic", return_value=fake_client):
            service = ClaudeSummarizerService()
            result = service.summarize_video(make_video(), make_long_transcript(68.0))

        # 5 chunk calls (ceil(68/15)) + 1 synthesis call
        assert call_count["n"] == 6
        assert len(result.key_points) == 5
        assert result.raw_summary == "Merged overview."

    def test_short_video_makes_exactly_one_call(self):
        fake_message = MagicMock()
        fake_message.content = [MagicMock(text=json.dumps({
            "key_points": ["a", "b", "c"],
            "raw_summary": "Fake summary.",
        }))]
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_message

        with patch("youtube_summarizer.services.summarizer_service.anthropic.Anthropic", return_value=fake_client):
            service = ClaudeSummarizerService()
            service.summarize_video(make_video(), make_long_transcript(10.0))

        assert fake_client.messages.create.call_count == 1

    def test_one_bad_chunk_does_not_sink_the_whole_video(self):
        """
        A single chunk returning unparseable JSON used to crash the whole
        video's summarization. It should now just be dropped, keeping the
        results from every chunk that succeeded.
        """
        call_count = {"n": 0}

        def fake_create(**kwargs):
            call_count["n"] += 1
            message = MagicMock()
            if "merging summaries" in kwargs["messages"][0]["content"]:
                message.content = [MagicMock(text=json.dumps({"raw_summary": "Merged overview."}))]
            elif call_count["n"] == 2:
                message.content = [MagicMock(text="not valid json at all")]
            else:
                message.content = [MagicMock(text=json.dumps({
                    "key_points": [f"point from call {call_count['n']}"],
                    "raw_summary": f"Chunk {call_count['n']} summary.",
                }))]
            return message

        fake_client = MagicMock()
        fake_client.messages.create.side_effect = fake_create

        with patch("youtube_summarizer.services.summarizer_service.anthropic.Anthropic", return_value=fake_client):
            service = ClaudeSummarizerService()
            result = service.summarize_video(make_video(), make_long_transcript(68.0))

        assert len(result.key_points) == 4  # 5 chunks, 1 failed
        assert result.raw_summary == "Merged overview."

    def test_all_chunks_failing_falls_back_to_single_pass(self):
        fake_message = MagicMock()
        fake_message.content = [MagicMock(text=json.dumps({
            "key_points": ["fallback a", "fallback b"],
            "raw_summary": "Fallback single-pass summary.",
        }))]
        fake_client = MagicMock()

        call_count = {"n": 0}

        def fake_create(**kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 5:  # every chunk call fails
                raise RuntimeError("simulated API failure")
            return fake_message

        fake_client.messages.create.side_effect = fake_create

        with patch("youtube_summarizer.services.summarizer_service.anthropic.Anthropic", return_value=fake_client):
            service = ClaudeSummarizerService()
            result = service.summarize_video(make_video(), make_long_transcript(68.0))

        assert result.key_points == ["fallback a", "fallback b"]
        assert result.raw_summary == "Fallback single-pass summary."

    def test_synthesis_failure_falls_back_to_first_chunk_summary(self):
        def fake_create(**kwargs):
            message = MagicMock()
            if "merging summaries" in kwargs["messages"][0]["content"]:
                message.content = [MagicMock(text="not valid json")]
            else:
                message.content = [MagicMock(text=json.dumps({
                    "key_points": ["a"],
                    "raw_summary": "First real chunk summary.",
                }))]
            return message

        fake_client = MagicMock()
        fake_client.messages.create.side_effect = fake_create

        with patch("youtube_summarizer.services.summarizer_service.anthropic.Anthropic", return_value=fake_client):
            service = ClaudeSummarizerService()
            result = service.summarize_video(make_video(), make_long_transcript(68.0))

        assert result.raw_summary == "First real chunk summary."


class TestOpenAIChunkedSummarization:

    def test_long_video_triggers_multiple_calls_and_merges_results(self):
        call_count = {"n": 0}

        def fake_create(**kwargs):
            call_count["n"] += 1
            response = MagicMock()
            content = kwargs["messages"][0]["content"]
            if "merging summaries" in content:
                payload = {"raw_summary": "Merged overview."}
            else:
                marker = first_marker_in(content)
                payload = {
                    "key_points": [{"text": f"point from call {call_count['n']}", "timestamp_marker": marker}],
                    "raw_summary": f"Chunk {call_count['n']} summary.",
                    "categories": [{"category": "Topic", "percentage": 100}],
                }
            response.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]
            return response

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = fake_create

        with patch("youtube_summarizer.services.openai_summarizer_service.OpenAI", return_value=fake_client):
            service = OpenAISummarizerService()
            result = service.summarize_video(make_video(), make_long_transcript(68.0))

        assert call_count["n"] == 6  # 5 chunks + 1 synthesis
        assert len(result.key_points_timed) == 5
        assert result.raw_summary == "Merged overview."
        assert len(result.categories) == 1
        assert result.categories[0].percentage == 100
        # Markers genuinely present in each chunk must survive validation
        # and convert correctly — none should be dropped as invalid.
        assert all(kp.timestamp is not None for kp in result.key_points_timed)

    def test_short_video_makes_exactly_one_call(self):
        fake_response = MagicMock()
        fake_response.choices = [MagicMock(message=MagicMock(content=json.dumps({
            "key_points": [{"text": "a", "timestamp_marker": "00:00"}],
            "raw_summary": "Fake summary.",
            "categories": [],
        })))]
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        with patch("youtube_summarizer.services.openai_summarizer_service.OpenAI", return_value=fake_client):
            service = OpenAISummarizerService()
            service.summarize_video(make_video(), make_long_transcript(10.0))

        assert fake_client.chat.completions.create.call_count == 1

    def test_fabricated_marker_from_wrong_time_frame_is_rejected(self):
        """
        Reproduces the real bug: for a chunk covering minutes 45-60 of the
        video, the model returned a well-formed but WRONG marker "00:45" —
        treating the excerpt as if it started its own clock at 0:00,
        instead of copying a real marker like "45:xx" from the text. That
        marker doesn't appear anywhere in that chunk's own transcript, so
        it must be rejected (timestamp -> None) rather than trusted.
        """
        def fake_create(**kwargs):
            response = MagicMock()
            content = kwargs["messages"][0]["content"]
            if "merging summaries" in content:
                payload = {"raw_summary": "Merged overview."}
            elif "[45:" in content:  # the chunk covering minutes 45-60
                payload = {
                    "key_points": [{"text": "fabricated timestamp", "timestamp_marker": "00:45"}],
                    "raw_summary": "Chunk summary.",
                    "categories": [],
                }
            else:
                marker = first_marker_in(content)
                payload = {
                    "key_points": [{"text": "real point", "timestamp_marker": marker}],
                    "raw_summary": "Chunk summary.",
                    "categories": [],
                }
            response.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]
            return response

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = fake_create

        with patch("youtube_summarizer.services.openai_summarizer_service.OpenAI", return_value=fake_client):
            service = OpenAISummarizerService()
            result = service.summarize_video(make_video(), make_long_transcript(68.0))

        fabricated = [kp for kp in result.key_points_timed if kp.text == "fabricated timestamp"]
        assert len(fabricated) == 1
        assert fabricated[0].timestamp is None
        # Every other chunk's real, in-range marker should still survive.
        real_points = [kp for kp in result.key_points_timed if kp.text == "real point"]
        assert all(kp.timestamp is not None for kp in real_points)

    def test_one_bad_chunk_does_not_sink_the_whole_video(self):
        """
        A single chunk returning unparseable JSON used to crash the whole
        video's summarization. It should now just be dropped, keeping the
        results (and category weighting) from every chunk that succeeded.
        """
        call_count = {"n": 0}

        def fake_create(**kwargs):
            call_count["n"] += 1
            response = MagicMock()
            content = kwargs["messages"][0]["content"]
            if "merging summaries" in content:
                response.choices = [MagicMock(message=MagicMock(
                    content=json.dumps({"raw_summary": "Merged overview."})
                ))]
            elif call_count["n"] == 2:
                response.choices = [MagicMock(message=MagicMock(content="not valid json"))]
            else:
                marker = first_marker_in(content)
                payload = {
                    "key_points": [{"text": f"point from call {call_count['n']}", "timestamp_marker": marker}],
                    "raw_summary": f"Chunk {call_count['n']} summary.",
                    "categories": [{"category": "Topic", "percentage": 100}],
                }
                response.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]
            return response

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = fake_create

        with patch("youtube_summarizer.services.openai_summarizer_service.OpenAI", return_value=fake_client):
            service = OpenAISummarizerService()
            result = service.summarize_video(make_video(), make_long_transcript(68.0))

        assert len(result.key_points_timed) == 4  # 5 chunks, 1 failed
        assert result.raw_summary == "Merged overview."

    def test_all_chunks_failing_falls_back_to_single_pass(self):
        fake_response = MagicMock()
        fake_response.choices = [MagicMock(message=MagicMock(content=json.dumps({
            "key_points": [{"text": "fallback", "timestamp_marker": "00:00"}],
            "raw_summary": "Fallback single-pass summary.",
            "categories": [],
        })))]

        call_count = {"n": 0}

        def fake_create(**kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 5:  # every chunk call fails
                raise RuntimeError("simulated API failure")
            return fake_response

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = fake_create

        with patch("youtube_summarizer.services.openai_summarizer_service.OpenAI", return_value=fake_client):
            service = OpenAISummarizerService()
            result = service.summarize_video(make_video(), make_long_transcript(68.0))

        assert result.key_points == ["fallback"]
        assert result.raw_summary == "Fallback single-pass summary."
