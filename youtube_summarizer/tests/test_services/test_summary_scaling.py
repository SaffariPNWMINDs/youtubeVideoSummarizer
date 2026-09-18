import pytest

from youtube_summarizer.services.summary_scaling import (
    key_point_count_for_duration,
    scale_max_tokens,
)


class TestKeyPointCountForDuration:

    @pytest.mark.parametrize(
        "minutes,expected_count",
        [
            (2, 3),        # very short clip
            (5, 3),        # boundary — still in the shortest tier
            (10, 5),       # typical tutorial length — matches old fixed default
            (15, 5),       # boundary
            (21, 7),       # the case from the bug report
            (60, 9),       # boundary
            (90, 11),      # boundary
            (120, 13),     # the 2-hour case from the bug report
            (300, 13),     # very long — capped, doesn't grow unbounded
        ],
    )
    def test_scales_with_duration(self, minutes, expected_count):
        assert key_point_count_for_duration(minutes * 60) == expected_count

    def test_zero_duration_gets_shortest_tier(self):
        assert key_point_count_for_duration(0) == 3


class TestScaleMaxTokens:

    def test_baseline_count_returns_base_unchanged(self):
        assert scale_max_tokens(600, 5) == 600

    def test_more_key_points_increases_budget(self):
        assert scale_max_tokens(600, 13) > scale_max_tokens(600, 5)

    def test_fewer_than_baseline_does_not_shrink_budget(self):
        # A short video asking for only 3 points shouldn't get a smaller
        # budget than what the base was tuned for.
        assert scale_max_tokens(600, 3) == 600
