

from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
from fastapi import HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from youtube_summarizer import factory
from youtube_summarizer.pipeline.rag_pipeline import retrieve_relevant_chunks, chunk_transcript, embed_texts, store_embeddings, is_video_indexed


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
    provider: Optional[LLMProvider] = LLMProvider.openai
    published_after_year: Optional[int] = None   # e.g. 2020 → only videos from 2020+
    duration: Optional[str] = None               # "short" | "medium" | "long"
    min_views: Optional[int] = None              # e.g. 10000
    
class KeyPointResponse(BaseModel):
    text: str
    timestamp: Optional[int] = None

class CategoryResponse(BaseModel):
    category: str
    percentage: int

class VideoResponse(BaseModel):
    video_id: str
    title: str
    channel_name: str
    view_count: int
    video_url: str
    key_points: List[str]
    key_points_timed: List[KeyPointResponse] = []
    raw_summary: str
    categories: List[CategoryResponse] = []

class SummarizeResponse(BaseModel):
    query: str
    videos: List[VideoResponse]
    key_takeaways: List[str]        # 5-7 top insights across all videos
    final_summary: str              # cross-video synthesis
    

class SummarizeUrlsRequest(BaseModel):
    urls: List[str]
    provider: Optional[LLMProvider] = LLMProvider.openai

@app.post("/summarize-urls/stream")
def summarize_urls_stream(request: SummarizeUrlsRequest):
    pipeline = factory.build_pipeline(len(request.urls), request.provider)
    return StreamingResponse(
        pipeline.stream_from_urls(request.urls),
        media_type="application/x-ndjson"
    )


class AskRequest(BaseModel):
    video_id: str
    transcript: str
    question: str

class AskResponse(BaseModel):
    answer: str
    relevant_chunks: List[str]

@app.post("/ask")
def ask(request: AskRequest):
    if not is_video_indexed(request.video_id):
        chunks = chunk_transcript(request.transcript)
        embeddings = embed_texts(chunks)
        store_embeddings(request.video_id, chunks, embeddings)
    relevant_chunks = retrieve_relevant_chunks(request.video_id, request.question)

    context = "\n\n".join(relevant_chunks)
    from openai import OpenAI
    llm = OpenAI()

    def stream_answer():
        for chunk in llm.chat.completions.create(
            model="gpt-4o-mini",
            stream=True,
            messages=[
                {"role": "system", "content": "Answer the question using only the provided transcript context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"},
            ],
        ):
            token = chunk.choices[0].delta.content
            if token:
                yield token

    return StreamingResponse(stream_answer(), media_type="text/plain")


@app.post("/summarize/stream")
def summarize_stream(request: SummarizeRequest):
    pipeline = factory.build_pipeline(request.max_videos, request.provider)
    return StreamingResponse(
        pipeline.stream(
            request.query,
            published_after_year=request.published_after_year,
            duration=request.duration,
            min_views=request.min_views,
        ),
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
                key_points_timed=[KeyPointResponse(text=kp.text, timestamp=kp.timestamp) for kp in vs.key_points_timed],
                raw_summary=vs.raw_summary,
                categories=[CategoryResponse(category=c.category, percentage=c.percentage) for c in vs.categories]
            ) for vs in result.video_summaries
        ]
    )
