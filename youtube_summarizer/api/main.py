

from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
from fastapi import HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from youtube_summarizer import factory


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class LLMProvider(str, Enum):
    openai = "openai"
    claude = "claude"

class SummarizeRequest(BaseModel):
    query: str
    max_videos: Optional[int] = 5
    provider: Optional[LLMProvider] = LLMProvider.openai  # or LLMProvider.claude
    
class VideoResponse(BaseModel):
    video_id: str
    title: str
    channel_name: str
    view_count: int
    video_url: str
    key_points: List[str]       # 3-5 bullet points
    raw_summary: str            # 2-3 sentence prose summary

class SummarizeResponse(BaseModel):
    query: str
    videos: List[VideoResponse]
    key_takeaways: List[str]        # 5-7 top insights across all videos
    final_summary: str              # cross-video synthesis
    

@app.post("/summarize/stream")
def summarize_stream(request: SummarizeRequest):
    pipeline = factory.build_pipeline(request.max_videos, request.provider)
    return StreamingResponse(
        pipeline.stream(request.query),
        media_type="application/x-ndjson"
    )


@app.post("/summarize")
def summarize(request: SummarizeRequest):
    
    pipeline = factory.build_pipeline(request.max_videos, request.provider)
    result = pipeline.run(request.query)

    if result is None:
        # In APIs we raise HTTP errors, not sys.exit
        raise HTTPException(status_code=500, detail="Could not generate summary")

    return SummarizeResponse(
        query=result.query,
        final_summary=result.final_summary,
        key_takeaways=result.key_takeaways,
        videos = [
            VideoResponse(
                video_id=vs.video_id,
                title=vs.title,
                channel_name=vs.channel_name,
                view_count=vs.view_count,
                video_url=vs.video_url,
                key_points=vs.key_points,
                raw_summary=vs.raw_summary
            ) for vs in result.video_summaries
        ]
    )
