import json
import logging
from typing import List

import anthropic

from youtube_summarizer.config import settings
from youtube_summarizer.models.summary import AggregatedSummary, VideoSummary
from youtube_summarizer.models.transcript import Transcript
from youtube_summarizer.models.video import Video
from youtube_summarizer.services.base import BaseSummarizerService

logger = logging.getLogger(__name__)


class ClaudeSummarizerService(BaseSummarizerService):
    """
    Two-stage summarization strategy:
      Stage 1 — per_video_model (fast/cheap): summarize each video independently
      Stage 2 — aggregate_model (more capable): synthesize all summaries into
                 one coherent report

    This model-routing approach keeps costs low while producing high-quality
    final output. It's the map-reduce pattern applied to LLMs.
    """

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        logger.debug(
            f"ClaudeSummarizerService initialised "
            f"(per-video: {settings.per_video_model}, "
            f"aggregate: {settings.aggregate_model})"
        )

    # ------------------------------------------------------------------ #
    #  Public interface (implements BaseSummarizerService)                #
    # ------------------------------------------------------------------ #

    def summarize_video(self, video: Video, transcript: Transcript) -> VideoSummary:
        logger.info(f"Summarizing '{video.title[:55]}'")

        truncated_text = transcript.truncate(settings.max_transcript_chars)
        prompt = self._build_video_prompt(video, truncated_text)

        raw_response = self._call_llm(
            model=settings.per_video_model,
            prompt=prompt,
            max_tokens=600,
        )
        parsed = self._parse_json(raw_response)

        return VideoSummary(
            video_id=video.video_id,
            title=video.title,
            channel_name=video.channel_name,
            view_count=video.view_count,
            video_url=video.url,
            key_points=parsed["key_points"],
            raw_summary=parsed["raw_summary"],
        )

    def aggregate_summaries(
        self, query: str, summaries: List[VideoSummary]
    ) -> AggregatedSummary:
        logger.info(f"Aggregating {len(summaries)} summaries for '{query}'")

        prompt = self._build_aggregate_prompt(query, summaries)
        raw_response = self._call_llm(
            model=settings.aggregate_model,
            prompt=prompt,
            max_tokens=1200,
        )
        parsed = self._parse_json(raw_response)

        return AggregatedSummary(
            query=query,
            video_summaries=summaries,
            final_summary=parsed["final_summary"],
            key_takeaways=parsed["key_takeaways"],
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _call_llm(self, model: str, prompt: str, max_tokens: int) -> str:
        """Single LLM call — all API interaction goes through here."""
        message = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """
        Strips markdown code fences if present, then parses JSON.
        LLMs sometimes wrap JSON in ```json ... ``` even when told not to.
        """
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Drop first line (```json or ```) and last line (```)
            text = "\n".join(lines[1:-1]).strip()
        return json.loads(text)

    @staticmethod
    def _build_video_prompt(video: Video, transcript_text: str) -> str:
        return f"""You are summarizing a YouTube video transcript. Be concise and factual.

Video title: {video.title}
Channel: {video.channel_name}
Views: {video.view_count:,}

Transcript:
{transcript_text}

Return ONLY a JSON object (no markdown, no explanation) with these exact keys:
{{
  "key_points": ["point 1", "point 2", "point 3"],
  "raw_summary": "2-3 sentence prose summary of the main content."
}}

Rules:
- key_points: 3 to 5 items, each under 20 words, written as statements not questions
- raw_summary: plain prose, under 100 words
- JSON only — no markdown fences, no extra text"""

    @staticmethod
    def _build_aggregate_prompt(query: str, summaries: List[VideoSummary]) -> str:
        summaries_block = "\n\n".join(
            f"[Video {i+1}] {s.title} ({s.view_count:,} views)\n"
            + "\n".join(f"  • {p}" for p in s.key_points)
            for i, s in enumerate(summaries)
        )

        return f"""You are synthesizing insights from {len(summaries)} YouTube videos about: "{query}"

Individual video key points:
{summaries_block}

Return ONLY a JSON object (no markdown, no explanation) with these exact keys:
{{
  "final_summary": "3-4 sentence synthesis that captures the consensus and any notable disagreements across all videos.",
  "key_takeaways": ["takeaway 1", "takeaway 2", "takeaway 3", "takeaway 4", "takeaway 5"]
}}

Rules:
- final_summary: synthesize across ALL videos, mention where they agree or diverge
- key_takeaways: 5 to 7 items, most important insights, ordered by importance
- JSON only — no markdown fences, no extra text"""
