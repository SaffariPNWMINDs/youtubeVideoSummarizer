"""
SearchPipeline — the orchestrator.

OOP lesson — Dependency Injection:
  The pipeline receives its services via the constructor rather than
  creating them internally. This is crucial for:
    - Testing: pass in Mock services, no real API calls needed
    - Flexibility: swap implementations without changing pipeline logic
    - Single Responsibility: the pipeline orchestrates; services do the work
"""

import json
import logging
from typing import Optional, Generator

from youtube_summarizer.models.summary import AggregatedSummary
from youtube_summarizer.services.base import (
    BaseTranscriptService,
    BaseSummarizerService,
    BaseVideoSearchService,
)

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._search = search_service
        self._transcript = transcript_service
        self._summarizer = summarizer_service
        self._max_videos = max_videos
        self._min_summaries = min_summaries

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

        # --- Steps 2 + 3: Fetch transcripts and summarize ---
        video_summaries = []
        skipped = 0

        for i, video in enumerate(videos, 1):
            logger.info(f"Processing video {i}/{len(videos)}: {video.title[:50]}")

            transcript = self._transcript.fetch(video)
            if transcript is None:
                skipped += 1
                logger.warning("  Skipped (no transcript)")
                continue

            try:
                summary = self._summarizer.summarize_video(video, transcript)
                video_summaries.append(summary)
                logger.info(f"  ✓ Summarized ({len(summary.key_points)} key points)")
            except Exception as e:
                skipped += 1
                logger.error(f"  ✗ Summarization failed: {e}")

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

    def stream(self, query: str) -> Generator[str, None, None]:
        """
        Same pipeline as run() but yields NDJSON chunks as work completes.
        Each yielded string is a JSON line the frontend can render immediately.
        """
        def emit(obj: dict) -> str:
            return json.dumps(obj) + "\n"

        yield emit({"type": "status", "message": "Searching for videos..."})

        videos = self._search.search(query, max_results=self._max_videos)
        if not videos:
            yield emit({"type": "error", "message": "No videos found"})
            return

        yield emit({"type": "status", "message": f"Found {len(videos)} videos. Summarizing..."})

        video_summaries = []
        skipped = 0

        for i, video in enumerate(videos, 1):
            yield emit({"type": "status", "message": f"Processing video {i}/{len(videos)}: {video.title[:50]}"})

            transcript = self._transcript.fetch(video)
            if transcript is None:
                skipped += 1
                continue

            try:
                summary = self._summarizer.summarize_video(video, transcript)
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
                        "raw_summary": summary.raw_summary,
                    }
                })
            except Exception as e:
                skipped += 1
                logger.error(f"Summarization failed: {e}")

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
