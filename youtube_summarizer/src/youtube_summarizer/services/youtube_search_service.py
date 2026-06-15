import logging
from datetime import datetime, timezone
from typing import List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from youtube_summarizer.config import settings
from youtube_summarizer.models.video import Video
from youtube_summarizer.services.base import BaseVideoSearchService

logger = logging.getLogger(__name__)


class YouTubeSearchService(BaseVideoSearchService):
    """
    Searches YouTube via the Data API v3.

    Two-step process:
      1. search.list  — finds video IDs matching the query
      2. videos.list  — fetches full stats (view count, etc.) for those IDs

    We sort by view count after fetching stats, since the search API's
    "order=viewCount" doesn't always return the most-viewed results.
    """

    def __init__(self) -> None:
        # Build the client once and reuse it — avoids re-discovery on every call
        self._client = build(
            "youtube", "v3", developerKey=settings.youtube_api_key
        )
        logger.debug("YouTubeSearchService initialised")

    def search(
        self,
        query: str,
        max_results: int = 10,
        published_after_year: Optional[int] = None,
        duration: Optional[str] = None,
        min_views: Optional[int] = None,
    ) -> List[Video]:
        logger.info(f"Searching YouTube for: '{query}' (max {max_results})")

        try:
            video_ids = self._search_video_ids(query, max_results, published_after_year, duration)
            if not video_ids:
                logger.warning("YouTube search returned no results")
                return []

            videos = self._fetch_video_details(video_ids)

            if min_views:
                videos = [v for v in videos if v.view_count >= min_views]

            videos.sort(key=lambda v: v.view_count, reverse=True)
            logger.info(f"Returning {len(videos)} videos for '{query}'")
            return videos

        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            raise

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _search_video_ids(
        self,
        query: str,
        max_results: int,
        published_after_year: Optional[int] = None,
        duration: Optional[str] = None,
    ) -> List[str]:
        """Step 1: get video IDs from search.list."""
        params = dict(
            q=query,
            part="id",
            maxResults=max_results,
            type="video",
            order="relevance",
            relevanceLanguage="en",
            safeSearch="moderate",
        )
        if published_after_year:
            params["publishedAfter"] = datetime(published_after_year, 1, 1, tzinfo=timezone.utc).isoformat()
        if duration in ("short", "medium", "long"):
            params["videoDuration"] = duration

        response = self._client.search().list(**params).execute()
        return [
            item["id"]["videoId"]
            for item in response.get("items", [])
            if item.get("id", {}).get("videoId")
        ]

    def _fetch_video_details(self, video_ids: List[str]) -> List[Video]:
        """Step 2: get full metadata + stats for a list of video IDs."""
        response = (
            self._client.videos()
            .list(
                part="snippet,statistics",
                id=",".join(video_ids),
            )
            .execute()
        )

        videos: List[Video] = []
        for item in response.get("items", []):
            try:
                video = Video(
                    video_id=item["id"],
                    title=item["snippet"]["title"],
                    channel_name=item["snippet"]["channelTitle"],
                    view_count=int(item["statistics"].get("viewCount", 0)),
                    published_at=item["snippet"]["publishedAt"],
                    description=item["snippet"].get("description"),
                )
                videos.append(video)
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed video item {item.get('id')}: {e}")

        return videos
