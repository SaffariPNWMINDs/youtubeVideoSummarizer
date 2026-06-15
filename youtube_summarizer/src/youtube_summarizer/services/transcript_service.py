import logging
from pathlib import Path
from typing import Optional

import requests

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from youtube_summarizer.models.transcript import Transcript, TranscriptSegment
from youtube_summarizer.models.video import Video
from youtube_summarizer.services.base import BaseTranscriptService

logger = logging.getLogger(__name__)

# Languages to try, in order of preference
_PREFERRED_LANGUAGES = ["en", "en-US", "en-GB", "en-CA", "en-AU"]


class YouTubeTranscriptService(BaseTranscriptService):
    """
    Fetches captions from YouTube using the youtube-transcript-api library.

    Graceful degradation strategy:
      1. Try English captions (manual or auto-generated)
      2. If not found → return None (pipeline will skip this video)
      3. Any other error → log and return None (never crash the pipeline)

    NOTE: In V1 we will add a Whisper fallback here for videos with no captions.
    The BaseTranscriptService interface won't change — just this implementation.
    """

    def fetch(self, video: Video) -> Optional[Transcript]:
        logger.info(f"Fetching transcript for '{video.title[:60]}'")

        try:
            cookie_path = Path(__file__).parent.parent.parent.parent / "youtube_cookies.txt"
            session = requests.Session()
            if cookie_path.exists():
                import http.cookiejar
                jar = http.cookiejar.MozillaCookieJar(str(cookie_path))
                jar.load()
                session.cookies = jar
            ytt_api = YouTubeTranscriptApi(http_client=session)
            raw_segments = ytt_api.fetch(
                video.video_id,
                languages=_PREFERRED_LANGUAGES,
            ).to_raw_data()
            transcript = self._build_transcript(video.video_id, raw_segments)
            logger.info(
                f"  ✓ {transcript.word_count:,} words "
                f"({transcript.char_count:,} chars)"
            )
            return transcript

        except (TranscriptsDisabled, NoTranscriptFound):
            logger.warning(f"  ✗ No transcript available for {video.video_id}")
            return None

        except VideoUnavailable:
            logger.warning(f"  ✗ Video {video.video_id} is unavailable")
            return None

        except Exception as e:
            # Catch-all: never let a single video crash the whole pipeline
            logger.error(f"  ✗ Unexpected error for {video.video_id}: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_transcript(video_id: str, raw: list) -> Transcript:
        """Convert the raw API list-of-dicts into our typed Transcript model."""
        segments = [
            TranscriptSegment(
                text=seg["text"].strip(),
                start=float(seg["start"]),
                duration=float(seg["duration"]),
            )
            for seg in raw
            if seg.get("text", "").strip()          # skip empty segments
        ]
        return Transcript(video_id=video_id, segments=segments, language="en")
