"""
Splits a long transcript into time-ordered chunks so each chunk's context
is small enough for the LLM to reliably attend to.

A single call asked to summarize a 60+ minute transcript in one shot shows
a well-documented "lost in the middle" bias — it concentrates key points
near the start of the transcript and effectively ignores the rest, even
though the full text technically fits in the model's context window.
Chunking, summarizing each piece independently, and merging the results is
the same map-reduce idea this project already applies across videos —
just applied within one long video's own transcript.
"""

from typing import List

from youtube_summarizer.models.summary import CategoryBreakdown
from youtube_summarizer.models.transcript import Transcript, TranscriptSegment

# Videos at or under this length are summarized in a single call — no
# behavior change from the non-chunked path. Only genuinely long videos
# pay the extra cost/latency of multiple calls.
CHUNKING_THRESHOLD_SECONDS = 20 * 60

# Each chunk covers roughly this much of the video: small enough that the
# model reliably attends to the whole chunk, big enough to keep the call
# count (and cost) reasonable.
CHUNK_MINUTES = 15.0


def needs_chunking(duration_seconds: float) -> bool:
    return duration_seconds > CHUNKING_THRESHOLD_SECONDS


def chunk_transcript(transcript: Transcript, chunk_minutes: float = CHUNK_MINUTES) -> List[Transcript]:
    """
    Splits into consecutive time windows. A chunk boundary always falls
    between segments, never inside one. Segment start times stay absolute
    (relative to the full video), so key point timestamps produced from any
    chunk remain valid on the original video without adjustment.
    """
    if not transcript.segments:
        return [transcript]

    chunk_seconds = chunk_minutes * 60
    chunks: List[Transcript] = []
    current: List[TranscriptSegment] = []
    window_start = transcript.segments[0].start

    for seg in transcript.segments:
        if current and (seg.start - window_start) >= chunk_seconds:
            chunks.append(Transcript(video_id=transcript.video_id, segments=current, language=transcript.language))
            current = []
            window_start = seg.start
        current.append(seg)

    if current:
        chunks.append(Transcript(video_id=transcript.video_id, segments=current, language=transcript.language))

    return chunks


def allocate_key_points(total_key_points: int, chunk_durations: List[float]) -> List[int]:
    """
    Distributes the video's total key point target across chunks,
    weighted by each chunk's share of the total duration. Every chunk
    gets at least 1, so short trailing chunks still contribute something.
    """
    total_duration = sum(chunk_durations) or 1.0
    return [max(1, round(total_key_points * d / total_duration)) for d in chunk_durations]


def merge_categories(
    chunk_categories: List[List[CategoryBreakdown]],
    weights: List[float],
    max_categories: int = 6,
) -> List[CategoryBreakdown]:
    """
    Merges per-chunk category breakdowns into one, weighting each chunk's
    percentages by its share of total video duration — so a topic that
    dominates a short chunk doesn't get equal say to one spanning a long
    chunk. Categories are matched by name (case-insensitive, trimmed).
    """
    totals: dict[str, float] = {}
    total_weight = sum(weights) or 1.0

    for categories, weight in zip(chunk_categories, weights):
        share = weight / total_weight
        for c in categories:
            key = c.category.strip()
            totals[key] = totals.get(key, 0.0) + c.percentage * share

    if not totals:
        return []

    total_pct = sum(totals.values()) or 1.0
    merged = sorted(
        (
            CategoryBreakdown(category=name, percentage=round(100 * pct / total_pct))
            for name, pct in totals.items()
        ),
        key=lambda c: c.percentage,
        reverse=True,
    )
    return merged[:max_categories]


def build_chunk_synthesis_prompt(video_title: str, chunk_summaries: List[str]) -> str:
    """
    Shared, provider-agnostic prompt for merging several chunk-level
    summaries of one long video into a single overall summary.
    """
    joined = "\n".join(f"- {s}" for s in chunk_summaries)
    return f"""You are merging summaries of consecutive segments of one long YouTube video into a single overview.

Video title: {video_title}

Segment summaries, in chronological order:
{joined}

Return ONLY a JSON object (no markdown, no explanation) with this exact key:
{{"raw_summary": "2-3 sentence prose summary covering the full video end to end, under 100 words."}}

JSON only — no markdown fences, no extra text."""
