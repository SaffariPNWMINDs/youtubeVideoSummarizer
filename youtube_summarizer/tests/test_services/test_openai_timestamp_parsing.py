"""
Confirms timestamps are computed deterministically in code from a copied
[MM:SS] marker, instead of asking the LLM to do the MM->seconds arithmetic
itself. The model was occasionally returning the raw minute number (e.g.
45) instead of the actual elapsed seconds (2705) for content near
[45:05] — a copy-the-marker instruction plus code-side parsing removes
that failure mode entirely, since Python's arithmetic never gets it wrong.

Bare numbers (no colon) are rejected rather than accepted as "already
seconds" — a bare 45 is exactly what the original bug looked like, and
there's no way to tell it apart from a correct raw-seconds value, so
guessing would silently let the bug back in.
"""

import pytest

from youtube_summarizer.services.openai_summarizer_service import OpenAISummarizerService


class TestParseTimestampMarker:

    @pytest.mark.parametrize(
        "marker,expected_seconds",
        [
            ("00:45", 45),
            ("02:14", 134),
            ("45:05", 2705),   # the exact case from the bug report
            ("75:07", 4507),
            ("00:00", 0),
        ],
    )
    def test_converts_mm_ss_marker_to_seconds(self, marker, expected_seconds):
        assert OpenAISummarizerService._parse_timestamp_marker(marker) == expected_seconds

    def test_none_marker_returns_none(self):
        assert OpenAISummarizerService._parse_timestamp_marker(None) is None

    def test_malformed_marker_returns_none(self):
        assert OpenAISummarizerService._parse_timestamp_marker("not a marker") is None

    def test_bracketed_marker_is_normalized_before_matching(self):
        # The model sometimes copies "[16:34]" including the brackets,
        # rather than just "16:34" — this is otherwise a real, correct
        # marker and shouldn't be rejected over formatting alone.
        assert OpenAISummarizerService._parse_timestamp_marker(
            "[16:34]", valid_markers={"16:34"}
        ) == 994

    def test_marker_not_present_in_this_chunk_is_rejected(self):
        # Reproduces the real bug: a well-formed marker that simply isn't
        # one of the markers that actually appear in this chunk's text —
        # e.g. "00:45" fabricated for a chunk covering minutes 45-60.
        assert OpenAISummarizerService._parse_timestamp_marker(
            "00:45", valid_markers={"45:05", "46:10"}
        ) is None

    def test_bare_number_with_no_colon_is_rejected_not_guessed(self):
        # This is exactly the shape of the original bug: the model writing
        # a bare minute number instead of a proper MM:SS marker. There's
        # no way to tell that apart from a genuinely correct raw-seconds
        # value, so it must be dropped rather than trusted.
        assert OpenAISummarizerService._parse_timestamp_marker(45) is None
        assert OpenAISummarizerService._parse_timestamp_marker("134") is None
