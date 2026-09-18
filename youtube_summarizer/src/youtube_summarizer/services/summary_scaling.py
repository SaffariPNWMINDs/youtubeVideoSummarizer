"""
Scales per-video summarization parameters to the video's actual length.

A 5-minute video and a 2-hour video cover very different amounts of
ground — asking both for the same fixed number of key points either
pads the short one with filler or crushes the long one down to almost
nothing useful. This maps video duration to a target key point count,
and the LLM's max_tokens budget to that count (more key points need
more room in the JSON response, or it gets truncated mid-object).
"""

# (duration upper bound in minutes, key point count) — first match wins.
# Tuned so the common 5-15 min case keeps today's default of 5.
_KEY_POINT_TIERS = [
    (5, 3),
    (15, 5),
    (30, 7),
    (60, 9),
    (90, 11),
    (float("inf"), 13),
]

_BASELINE_KEY_POINT_COUNT = 5     # what base max_tokens values were tuned for
_TOKENS_PER_EXTRA_KEY_POINT = 70  # rough JSON + prose cost per extra point


def key_point_count_for_duration(duration_seconds: float) -> int:
    """Longer videos cover more ground, so they earn more key points."""
    minutes = duration_seconds / 60
    for upper_bound_minutes, count in _KEY_POINT_TIERS:
        if minutes <= upper_bound_minutes:
            return count
    return _KEY_POINT_TIERS[-1][1]


def scale_max_tokens(base_max_tokens: int, key_point_count: int) -> int:
    """More key points need more room in the JSON response."""
    extra_points = max(0, key_point_count - _BASELINE_KEY_POINT_COUNT)
    return base_max_tokens + extra_points * _TOKENS_PER_EXTRA_KEY_POINT
