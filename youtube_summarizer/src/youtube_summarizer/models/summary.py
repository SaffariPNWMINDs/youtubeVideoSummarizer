from pydantic import BaseModel, Field
from typing import List
from datetime import datetime


class VideoSummary(BaseModel):
    """Summary produced for a single video."""

    video_id: str
    title: str
    channel_name: str
    view_count: int
    video_url: str
    key_points: List[str]       # 3-5 bullet points
    raw_summary: str            # 2-3 sentence prose summary


class AggregatedSummary(BaseModel):
    """
    Final output of the full pipeline.
    Combines insights from all successfully summarized videos.
    """

    query: str
    video_summaries: List[VideoSummary]
    final_summary: str              # cross-video synthesis
    key_takeaways: List[str]        # 5-7 top insights across all videos
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def video_count(self) -> int:
        return len(self.video_summaries)

    def to_markdown(self) -> str:
        """Renders the summary as clean markdown — useful for export later."""
        lines = [
            f"# Summary: {self.query}",
            f"*{self.video_count} videos analyzed · {self.created_at.strftime('%Y-%m-%d %H:%M')}*",
            "",
            "## Overview",
            self.final_summary,
            "",
            "## Key Takeaways",
        ]
        for i, takeaway in enumerate(self.key_takeaways, 1):
            lines.append(f"{i}. {takeaway}")

        lines += ["", "## Sources"]
        for vs in self.video_summaries:
            lines.append(f"- [{vs.title}]({vs.video_url}) — {vs.view_count:,} views")

        return "\n".join(lines)
