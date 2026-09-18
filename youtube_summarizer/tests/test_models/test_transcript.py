from youtube_summarizer.models.transcript import Transcript, TranscriptSegment


class TestTranscriptDuration:

    def test_duration_is_end_of_last_segment(self):
        transcript = Transcript(
            video_id="v1",
            segments=[
                TranscriptSegment(text="a", start=0.0, duration=5.0),
                TranscriptSegment(text="b", start=5.0, duration=10.0),
                TranscriptSegment(text="c", start=15.0, duration=3.0),
            ],
        )
        assert transcript.duration_seconds == 18.0

    def test_duration_uses_max_end_even_if_segments_are_out_of_order(self):
        transcript = Transcript(
            video_id="v1",
            segments=[
                TranscriptSegment(text="b", start=5.0, duration=10.0),
                TranscriptSegment(text="a", start=0.0, duration=5.0),
            ],
        )
        assert transcript.duration_seconds == 15.0

    def test_duration_is_zero_for_empty_transcript(self):
        transcript = Transcript(video_id="v1", segments=[])
        assert transcript.duration_seconds == 0.0


def make_long_transcript(total_chars: int, span_seconds: float = 4080.0) -> Transcript:
    """A single-segment transcript of exactly `total_chars` characters."""
    return Transcript(
        video_id="v1",
        segments=[TranscriptSegment(text="a" * total_chars, start=0.0, duration=span_seconds)],
    )


class TestTruncate:

    def test_returns_full_text_when_under_the_limit(self):
        transcript = make_long_transcript(100)
        assert transcript.truncate(max_chars=1000) == "a" * 100

    def test_cuts_content_past_the_limit(self):
        transcript = make_long_transcript(1000)
        result = transcript.truncate(max_chars=100)
        assert len(result) <= 100

    def test_logs_a_warning_when_content_is_actually_dropped(self, caplog):
        transcript = make_long_transcript(1000)
        with caplog.at_level("WARNING"):
            transcript.truncate(max_chars=100)
        assert any("truncated" in record.message for record in caplog.records)

    def test_does_not_log_when_nothing_is_dropped(self, caplog):
        transcript = make_long_transcript(100)
        with caplog.at_level("WARNING"):
            transcript.truncate(max_chars=1000)
        assert not caplog.records


class TestTruncateWithTimestamps:

    def test_logs_a_warning_when_content_is_actually_dropped(self, caplog):
        transcript = make_long_transcript(1000)
        with caplog.at_level("WARNING"):
            transcript.truncate_with_timestamps(max_chars=100)
        assert any("truncated" in record.message for record in caplog.records)
