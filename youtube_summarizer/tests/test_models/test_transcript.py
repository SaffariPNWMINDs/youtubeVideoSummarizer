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
