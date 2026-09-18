"""
SearchPipeline — the orchestrator.

Dependency Injection:
  The pipeline receives its services via the constructor rather than
  creating them internally. This is crucial for:
    - Testing: pass in Mock services, no real API calls needed
    - Flexibility: swap implementations without changing pipeline logic
    - Single Responsibility: the pipeline orchestrates; services do the work
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Generator, List, Tuple

from youtube_summarizer.config import settings
from youtube_summarizer.models.summary import AggregatedSummary, VideoSummary
from youtube_summarizer.models.transcript import Transcript
from youtube_summarizer.models.video import Video
from youtube_summarizer.services.base import (
    BaseTranscriptService,
    BaseSummarizerService,
    BaseVideoSearchService,
)

logger = logging.getLogger(__name__)


@dataclass
class _MapResult:
    """Outcome of running the map stage (fetch transcript + summarize) for one video."""
    transcript: Optional[Transcript]
    summary: Optional[VideoSummary]
    error: Optional[Exception]


class SearchPipeline:
    """
    Orchestrates: Search → Fetch Transcripts → Summarize → Aggregate.

    Usage:
        pipeline = SearchPipeline(
            search_service=YouTubeSearchService(),
            transcript_service=YouTubeTranscriptService(),
            summarizer_service=ClaudeSummarizerService(),
        )
        result = pipeline.run("machine learning basics")
    """

    def __init__(
        self,
        search_service: BaseVideoSearchService,
        transcript_service: BaseTranscriptService,
        summarizer_service: BaseSummarizerService,
        max_videos: int = 10,
        min_summaries: int = 3,     # abort if we get fewer than this
        max_workers: Optional[int] = None,
    ) -> None:
        self._search = search_service
        self._transcript = transcript_service
        self._summarizer = summarizer_service
        self._max_videos = max_videos
        self._min_summaries = min_summaries
        self._max_workers = max_workers or settings.pipeline_max_workers

    # ------------------------------------------------------------------ #
    #  Map stage — run per-video work (fetch transcript + summarize) in    #
    #  parallel across a thread pool. Each video's work is independent,    #
    #  so this is the "map" half of the map-reduce pattern; aggregation    #
    #  ("reduce") still happens once, after every video is mapped.         #
    # ------------------------------------------------------------------ #

    def _process_video(self, video: Video) -> _MapResult:
        """Runs on a worker thread: fetch transcript, then summarize."""
        transcript = self._transcript.fetch(video)
        if transcript is None:
            return _MapResult(transcript=None, summary=None, error=None)

        try:
            summary = self._summarizer.summarize_video(video, transcript)
            return _MapResult(transcript=transcript, summary=summary, error=None)
        except Exception as e:
            return _MapResult(transcript=transcript, summary=None, error=e)

    def _map_videos(
        self, videos: List[Video]
    ) -> Generator[Tuple[int, Video, _MapResult], None, None]:
        """
        Submits all videos to a thread pool at once, so transcript fetching
        and summarization happen concurrently, then yields results one by
        one in the *original* video order (each yield blocks only on that
        video's own future, which is typically already done or close to it
        since every future started running as soon as it was submitted).
        """
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [executor.submit(self._process_video, video) for video in videos]
            for i, (video, future) in enumerate(zip(videos, futures), 1):
                yield i, video, future.result()

    def run(self, query: str) -> Optional[AggregatedSummary]:
        """
        Run the full pipeline for a search query.
        Returns None if not enough videos could be summarized.
        """
        logger.info(f"{'='*55}")
        logger.info(f"Pipeline starting — query: '{query}'")
        logger.info(f"{'='*55}")

        # --- Step 1: Search ---
        videos = self._search.search(query, max_results=self._max_videos)
        if not videos:
            logger.warning("No videos found — aborting pipeline")
            return None
        logger.info(f"Step 1 complete: {len(videos)} videos found")

        # --- Steps 2 + 3: Fetch transcripts and summarize (parallel map stage) ---
        video_summaries = []
        skipped = 0

        for i, video, result in self._map_videos(videos):
            logger.info(f"Processing video {i}/{len(videos)}: {video.title[:50]}")

            if result.transcript is None:
                skipped += 1
                logger.warning("  Skipped (no transcript)")
                continue

            if result.error is not None:
                skipped += 1
                logger.error(f"  ✗ Summarization failed: {result.error}")
                continue

            video_summaries.append(result.summary)
            logger.info(f"  ✓ Summarized ({len(result.summary.key_points)} key points)")

        logger.info(
            f"Step 2+3 complete: {len(video_summaries)} summarized, {skipped} skipped"
        )

        # Abort if we don't have enough material to synthesize
        if len(video_summaries) < self._min_summaries:
            logger.error(
                f"Only {len(video_summaries)} summaries — "
                f"need at least {self._min_summaries}. Aborting."
            )
            return None

        # --- Step 4: Aggregate ---
        logger.info("Step 4: Aggregating all summaries...")
        result = self._summarizer.aggregate_summaries(query, video_summaries)
        logger.info("Pipeline complete ✓")
        return result

    def stream(
        self,
        query: str,
        published_after_year: Optional[int] = None,
        published_before_year: Optional[int] = None,
        sort_by: Optional[str] = "views",
        language: Optional[str] = "en",
        channel_filter: Optional[str] = None,
        exclude_keywords: Optional[str] = None,
        duration: Optional[str] = None,
        min_views: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Same pipeline as run() but yields NDJSON chunks as work completes.
        Each yielded string is a JSON line the frontend can render immediately.
        """
        def emit(obj: dict) -> str:
            return json.dumps(obj) + "\n"

        yield emit({"type": "status", "message": "Searching for videos..."})

        videos = self._search.search(
            query,
            max_results=self._max_videos,
            published_after_year=published_after_year,
            published_before_year=published_before_year,
            sort_by=sort_by,
            language=language,
            channel_filter=channel_filter,
            exclude_keywords=exclude_keywords,
            duration=duration,
            min_views=min_views,
        )
        if not videos:
            yield emit({"type": "error", "message": "No videos found"})
            return

        yield emit({"type": "status", "message": f"Found {len(videos)} videos. Summarizing..."})

        video_summaries = []
        skipped = 0

        for i, video, result in self._map_videos(videos):
            yield emit({"type": "status", "message": f"Processing video {i}/{len(videos)}: {video.title[:50]}"})

            if result.transcript is None:
                skipped += 1
                continue

            if result.error is not None:
                skipped += 1
                logger.error(f"Summarization failed: {result.error}")
                continue

            summary = result.summary
            video_summaries.append(summary)
            yield emit({
                "type": "video",
                "data": {
                    "video_id": summary.video_id,
                    "title": summary.title,
                    "channel_name": summary.channel_name,
                    "view_count": summary.view_count,
                    "video_url": summary.video_url,
                    "key_points": summary.key_points,
                    "key_points_timed": [{"text": kp.text, "timestamp": kp.timestamp} for kp in summary.key_points_timed],
                    "raw_summary": summary.raw_summary,
                    "categories": [{"category": c.category, "percentage": c.percentage} for c in summary.categories],
                    "transcript_text": result.transcript.full_text,
                }
            })

        if len(video_summaries) < self._min_summaries:
            yield emit({"type": "error", "message": f"Only {len(video_summaries)} videos could be summarized (need at least {self._min_summaries})"})
            return

        yield emit({"type": "status", "message": "Generating final summary..."})
        result = self._summarizer.aggregate_summaries(query, video_summaries)
        yield emit({
            "type": "final",
            "data": {
                "key_takeaways": result.key_takeaways,
                "final_summary": result.final_summary,
            }
        })

    def stream_from_urls(self, urls: List[str]) -> Generator[str, None, None]:
        """Skip search — summarize a user-provided list of YouTube URLs."""
        def emit(obj: dict) -> str:
            return json.dumps(obj) + "\n"

        video_ids = [_extract_video_id(u) for u in urls]
        video_ids = [vid for vid in video_ids if vid]

        if not video_ids:
            yield emit({"type": "error", "message": "No valid YouTube URLs found"})
            return

        yield emit({"type": "status", "message": f"Fetching metadata for {len(video_ids)} videos..."})
        videos = self._search.fetch_by_ids(video_ids)

        if not videos:
            yield emit({"type": "error", "message": "Could not fetch video metadata"})
            return

        yield emit({"type": "status", "message": f"Summarizing {len(videos)} videos..."})

        video_summaries = []
        skipped = 0

        for i, video, result in self._map_videos(videos):
            yield emit({"type": "status", "message": f"Processing video {i}/{len(videos)}: {video.title[:50]}"})

            if result.transcript is None:
                skipped += 1
                continue

            if result.error is not None:
                skipped += 1
                logger.error(f"Summarization failed: {result.error}")
                continue

            summary = result.summary
            video_summaries.append(summary)
            yield emit({
                "type": "video",
                "data": {
                    "video_id": summary.video_id,
                    "title": summary.title,
                    "channel_name": summary.channel_name,
                    "view_count": summary.view_count,
                    "video_url": summary.video_url,
                    "key_points": summary.key_points,
                    "key_points_timed": [{"text": kp.text, "timestamp": kp.timestamp} for kp in summary.key_points_timed],
                    "raw_summary": summary.raw_summary,
                    "categories": [{"category": c.category, "percentage": c.percentage} for c in summary.categories],
                    "transcript_text": result.transcript.full_text,
                }
            })

        if not video_summaries:
            yield emit({"type": "error", "message": "Could not summarize any of the provided videos"})
            return

        yield emit({"type": "status", "message": "Generating final summary..."})
        result = self._summarizer.aggregate_summaries("Provided videos", video_summaries)
        yield emit({
            "type": "final",
            "data": {
                "key_takeaways": result.key_takeaways,
                "final_summary": result.final_summary,
            }
        })


def _extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None
