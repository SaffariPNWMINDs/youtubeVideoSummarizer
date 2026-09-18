import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import anthropic

from youtube_summarizer.config import settings
from youtube_summarizer.models.summary import AggregatedSummary, VideoSummary
from youtube_summarizer.models.transcript import Transcript
from youtube_summarizer.models.video import Video
from youtube_summarizer.services.base import BaseSummarizerService
from youtube_summarizer.services.summary_scaling import (
    key_point_count_for_duration,
    scale_max_tokens,
)
from youtube_summarizer.services.transcript_chunking import (
    allocate_key_points,
    build_chunk_synthesis_prompt,
    chunk_transcript,
    needs_chunking,
)

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
            f"(per-video: {settings.claude_per_video_model}, "
            f"aggregate: {settings.claude_aggregate_model})"
        )

    # ------------------------------------------------------------------ #
    #  Public interface (implements BaseSummarizerService)                #
    # ------------------------------------------------------------------ #

    def summarize_video(self, video: Video, transcript: Transcript) -> VideoSummary:
        logger.info(f"Summarizing '{video.title[:55]}'")

        key_point_count = key_point_count_for_duration(transcript.duration_seconds)

        if needs_chunking(transcript.duration_seconds):
            return self._summarize_chunked(video, transcript, key_point_count)
        return self._summarize_single_pass(video, transcript, key_point_count)

    def _summarize_single_pass(
        self, video: Video, transcript: Transcript, key_point_count: int
    ) -> VideoSummary:
        truncated_text = transcript.truncate(settings.max_transcript_chars)
        parsed = self._summarize_text(video, truncated_text, key_point_count)

        return VideoSummary(
            video_id=video.video_id,
            title=video.title,
            channel_name=video.channel_name,
            view_count=video.view_count,
            video_url=video.url,
            key_points=parsed["key_points"],
            raw_summary=parsed["raw_summary"],
        )

    def _summarize_chunked(
        self, video: Video, transcript: Transcript, key_point_count: int
    ) -> VideoSummary:
        """
        Long videos get split into time-ordered chunks, each summarized
        independently (in parallel) so no single call has to attend to the
        entire transcript at once — see transcript_chunking.py for why.
        """
        chunks = chunk_transcript(transcript)
        chunk_durations = [c.duration_seconds for c in chunks]
        key_point_allocation = allocate_key_points(key_point_count, chunk_durations)
        logger.info(
            f"  Long video ({transcript.duration_seconds / 60:.0f} min) — "
            f"summarizing in {len(chunks)} chunks"
        )

        def summarize_chunk(chunk: Transcript, count: int) -> Optional[dict]:
            text = chunk.truncate(settings.max_transcript_chars)
            try:
                return self._summarize_text(video, text, count, is_excerpt=True)
            except Exception as e:
                # One bad chunk (malformed JSON, a dropped request, ...)
                # shouldn't sink the whole video — drop just this chunk's
                # contribution and keep the rest.
                logger.warning(f"  Chunk summarization failed, skipping it: {e}")
                return None

        with ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as executor:
            futures = [
                executor.submit(summarize_chunk, chunk, count)
                for chunk, count in zip(chunks, key_point_allocation)
            ]
            results = [r for r in (f.result() for f in futures) if r is not None]

        if not results:
            logger.warning("  All chunks failed — falling back to a single pass")
            return self._summarize_single_pass(video, transcript, key_point_count)

        all_key_points = [kp for r in results for kp in r["key_points"]]
        chunk_summaries = [r["raw_summary"] for r in results]
        final_summary = self._synthesize_summary(video, chunk_summaries)

        return VideoSummary(
            video_id=video.video_id,
            title=video.title,
            channel_name=video.channel_name,
            view_count=video.view_count,
            video_url=video.url,
            key_points=all_key_points,
            raw_summary=final_summary,
        )

    def _summarize_text(
        self, video: Video, transcript_text: str, key_point_count: int, is_excerpt: bool = False
    ) -> dict:
        prompt = self._build_video_prompt(video, transcript_text, key_point_count, is_excerpt)
        raw_response = self._call_llm(
            model=settings.claude_per_video_model,
            prompt=prompt,
            max_tokens=scale_max_tokens(600, key_point_count),
        )
        return self._parse_json(raw_response)

    def _synthesize_summary(self, video: Video, chunk_summaries: List[str]) -> str:
        try:
            prompt = build_chunk_synthesis_prompt(video.title, chunk_summaries)
            raw_response = self._call_llm(
                model=settings.claude_per_video_model, prompt=prompt, max_tokens=250
            )
            return self._parse_json(raw_response)["raw_summary"]
        except Exception as e:
            logger.warning(f"  Summary synthesis failed, using first chunk's summary: {e}")
            return chunk_summaries[0]

    def aggregate_summaries(
        self, query: str, summaries: List[VideoSummary]
    ) -> AggregatedSummary:
        logger.info(f"Aggregating {len(summaries)} summaries for '{query}'")

        prompt = self._build_aggregate_prompt(query, summaries)
        raw_response = self._call_llm(
            model=settings.claude_aggregate_model,
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
    def _build_video_prompt(
        video: Video, transcript_text: str, key_point_count: int, is_excerpt: bool = False
    ) -> str:
        subject = "one excerpt from a longer YouTube video's transcript" if is_excerpt else "a YouTube video transcript"
        breadth_note = (
            "Cover only what THIS EXCERPT discusses — you are not seeing the full video, "
            "so don't imply otherwise."
            if is_excerpt
            else "Cover the full breadth of what the video discusses — don't repeat the same idea in different words."
        )
        return f"""You are summarizing {subject}. Be concise and factual.

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
- key_points: exactly {key_point_count} items, each under 20 words, written as statements not questions. {breadth_note}
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
