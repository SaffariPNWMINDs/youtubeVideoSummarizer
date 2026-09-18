from youtube_summarizer.models.summary import CategoryBreakdown
from youtube_summarizer.models.transcript import Transcript, TranscriptSegment
from youtube_summarizer.services.transcript_chunking import (
    CHUNKING_THRESHOLD_SECONDS,
    allocate_key_points,
    chunk_transcript,
    merge_categories,
    needs_chunking,
)


def make_transcript(num_segments: int, segment_seconds: float = 10.0) -> Transcript:
    """A transcript made of many small, evenly-spaced segments."""
    return Transcript(
        video_id="v1",
        segments=[
            TranscriptSegment(text=f"segment {i}", start=i * segment_seconds, duration=segment_seconds)
            for i in range(num_segments)
        ],
    )


class TestNeedsChunking:

    def test_short_video_does_not_need_chunking(self):
        assert needs_chunking(10 * 60) is False

    def test_video_right_at_threshold_does_not_need_chunking(self):
        assert needs_chunking(CHUNKING_THRESHOLD_SECONDS) is False

    def test_long_video_needs_chunking(self):
        assert needs_chunking(68 * 60) is True


class TestChunkTranscript:

    def test_short_transcript_stays_one_chunk(self):
        transcript = make_transcript(num_segments=60, segment_seconds=10.0)  # 10 min total
        chunks = chunk_transcript(transcript, chunk_minutes=15.0)
        assert len(chunks) == 1
        assert chunks[0].duration_seconds == transcript.duration_seconds

    def test_long_transcript_splits_into_multiple_chunks(self):
        transcript = make_transcript(num_segments=68 * 6, segment_seconds=10.0)  # 68 min total
        chunks = chunk_transcript(transcript, chunk_minutes=15.0)
        assert len(chunks) == 5  # ceil(68/15)

    def test_chunks_cover_the_full_transcript_with_no_gaps_or_overlaps(self):
        transcript = make_transcript(num_segments=68 * 6, segment_seconds=10.0)
        chunks = chunk_transcript(transcript, chunk_minutes=15.0)

        all_texts_in_order = [seg.text for chunk in chunks for seg in chunk.segments]
        original_texts_in_order = [seg.text for seg in transcript.segments]
        assert all_texts_in_order == original_texts_in_order

    def test_segment_timestamps_stay_absolute_across_chunks(self):
        transcript = make_transcript(num_segments=68 * 6, segment_seconds=10.0)
        chunks = chunk_transcript(transcript, chunk_minutes=15.0)

        # The second chunk's first segment should NOT start back at 0 —
        # timestamps must stay relative to the whole video.
        assert chunks[1].segments[0].start > 0
        assert chunks[1].segments[0].start == chunks[0].segments[-1].end


class TestAllocateKeyPoints:

    def test_splits_proportionally_to_duration(self):
        allocation = allocate_key_points(total_key_points=10, chunk_durations=[60.0, 60.0])
        assert allocation == [5, 5]

    def test_every_chunk_gets_at_least_one(self):
        allocation = allocate_key_points(total_key_points=3, chunk_durations=[10.0, 1000.0])
        assert allocation[0] >= 1


class TestMergeCategories:

    def test_merges_matching_categories_weighted_by_duration(self):
        chunk_categories = [
            [CategoryBreakdown(category="AI", percentage=100)],
            [CategoryBreakdown(category="AI", percentage=0), CategoryBreakdown(category="Safety", percentage=100)],
        ]
        merged = merge_categories(chunk_categories, weights=[10.0, 10.0])
        names = {c.category for c in merged}
        assert names == {"AI", "Safety"}
        assert sum(c.percentage for c in merged) == 100

    def test_caps_at_max_categories(self):
        chunk_categories = [[CategoryBreakdown(category=f"Topic {i}", percentage=10) for i in range(10)]]
        merged = merge_categories(chunk_categories, weights=[10.0], max_categories=6)
        assert len(merged) == 6

    def test_empty_input_returns_empty(self):
        assert merge_categories([], []) == []
