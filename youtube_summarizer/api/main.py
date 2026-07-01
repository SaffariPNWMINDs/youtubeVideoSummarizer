

from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
from fastapi import HTTPException, FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build as google_build
from fastapi.responses import RedirectResponse
import json as _json

from youtube_summarizer import factory
from youtube_summarizer.config import settings
from youtube_summarizer.pipeline.rag_pipeline import retrieve_relevant_chunks, chunk_transcript, embed_texts, store_embeddings, is_video_indexed


app = FastAPI()

# In-memory store: video_id → full transcript text
transcript_store: dict[str, str] = {}

# In-memory store: session_id → user profile (interests + name)
user_sessions: dict[str, dict] = {}

# In-memory store: state → (flow, code_verifier) for OAuth PKCE
oauth_states: dict[str, dict] = {}

SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

def _build_flow() -> Flow:
    client_config = {
        "web": {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uris": [settings.google_redirect_uri],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    return Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=settings.google_redirect_uri)

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
    published_after_year: Optional[int] = None
    published_before_year: Optional[int] = None
    sort_by: Optional[str] = "views"
    language: Optional[str] = "en"
    channel_filter: Optional[str] = None
    exclude_keywords: Optional[str] = None
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
    

@app.get("/auth/login")
def auth_login():
    import os
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    flow = _build_flow()
    auth_url, state = flow.authorization_url(
        prompt="consent",
        access_type="offline",
    )
    oauth_states[state] = flow
    return {"auth_url": auth_url}


@app.get("/auth/callback")
def auth_callback(code: str, state: str = None):
    import os
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
    flow = oauth_states.pop(state, None) or _build_flow()
    flow.fetch_token(code=code)
    credentials = flow.credentials

    # Fetch user profile
    from googleapiclient.discovery import build as gbuild
    people_service = gbuild("oauth2", "v2", credentials=credentials)
    user_info = people_service.userinfo().get().execute()
    name = user_info.get("given_name", "there")
    email = user_info.get("email", "")

    # Fetch YouTube subscriptions (top 10)
    yt = gbuild("youtube", "v3", credentials=credentials)
    subs_response = yt.subscriptions().list(
        part="snippet", mine=True, maxResults=10
    ).execute()
    subscriptions = [
        item["snippet"]["title"]
        for item in subs_response.get("items", [])
    ]

    # Fetch liked videos (top 10)
    liked_response = yt.videos().list(
        part="snippet",
        myRating="like",
        maxResults=10,
    ).execute()
    liked_titles = [
        item["snippet"]["title"]
        for item in liked_response.get("items", [])
    ]

    # Generate personalized greeting via LLM
    from openai import OpenAI
    llm = OpenAI()
    context = f"User's name: {name}\nSubscribed channels: {', '.join(subscriptions)}\nRecently liked videos: {', '.join(liked_titles)}"
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a witty assistant. Generate a funny, personalized greeting for a user based on their YouTube interests. Maximum 20 words. Be creative and reference something specific. No emojis."},
            {"role": "user", "content": context},
        ],
        max_tokens=100,
    )
    greeting = response.choices[0].message.content

    # Store session
    session_id = email
    user_sessions[session_id] = {
        "name": name,
        "email": email,
        "subscriptions": subscriptions,
        "greeting": greeting,
    }

    # Redirect to frontend with session info
    frontend_url = f"http://localhost:5173?session={session_id}&name={name}&greeting={greeting}"
    return RedirectResponse(url=frontend_url)


@app.get("/auth/greeting")
def get_greeting(session: str):
    user = user_sessions.get(session)
    if not user:
        raise HTTPException(status_code=404, detail="Session not found")

    # Regenerate greeting each time for freshness
    from openai import OpenAI
    llm = OpenAI()
    context = f"User's name: {user['name']}\nSubscribed channels: {', '.join(user['subscriptions'])}"
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a witty assistant. Generate a funny, personalized greeting for a user based on their YouTube interests. Maximum 20 words. Be creative and reference something specific. No emojis."},
            {"role": "user", "content": context},
        ],
        max_tokens=100,
    )
    return {"greeting": response.choices[0].message.content, "name": user["name"]}


class SummarizeUrlsRequest(BaseModel):
    urls: List[str]
    provider: Optional[LLMProvider] = LLMProvider.openai

@app.post("/summarize-urls/stream")
def summarize_urls_stream(request: SummarizeUrlsRequest):
    pipeline = factory.build_pipeline(len(request.urls), request.provider)

    def stream_and_store():
        for chunk in pipeline.stream_from_urls(request.urls):
            import json as _json
            try:
                parsed = _json.loads(chunk.strip())
                if parsed.get("type") == "video":
                    vid = parsed["data"]
                    transcript_store[vid["video_id"]] = vid.pop("transcript_text", "")
            except Exception:
                pass
            yield chunk

    return StreamingResponse(stream_and_store(), media_type="application/x-ndjson")


@app.post("/summarize/stream")
def summarize_stream_endpoint(request: SummarizeRequest):
    pipeline = factory.build_pipeline(request.max_videos, request.provider)

    def stream_and_store():
        for chunk in pipeline.stream(
            request.query,
            published_after_year=request.published_after_year,
            published_before_year=request.published_before_year,
            sort_by=request.sort_by,
            language=request.language,
            channel_filter=request.channel_filter,
            exclude_keywords=request.exclude_keywords,
            duration=request.duration,
            min_views=request.min_views,
        ):
            import json as _json
            try:
                parsed = _json.loads(chunk.strip())
                if parsed.get("type") == "video":
                    vid = parsed["data"]
                    transcript_store[vid["video_id"]] = vid.pop("transcript_text", "")
            except Exception:
                pass
            yield chunk

    return StreamingResponse(stream_and_store(), media_type="application/x-ndjson")


class AskRequest(BaseModel):
    video_id: str
    question: str

class AskResponse(BaseModel):
    answer: str
    relevant_chunks: List[str]

@app.post("/ask")
def ask(request: AskRequest):
    if not is_video_indexed(request.video_id):
        transcript = transcript_store.get(request.video_id, "")
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found. Please search first.")
        chunks = chunk_transcript(transcript)
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



@app.post("/search-by-image/stream")
async def search_by_image_stream(
    image: UploadFile = File(...),
    description: str = Form(""),
    max_videos: int = Form(5),
    provider: str = Form("openai"),
):
    import base64
    from openai import OpenAI

    image_bytes = await image.read()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = image.content_type or "image/jpeg"

    llm = OpenAI()
    user_hint = f"\n\nUser's additional context: {description}" if description.strip() else ""
    vision_response = llm.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Look at this image carefully. Generate a concise, specific YouTube search query "
                            "(3-8 words) that would find the most informative and relevant videos about the "
                            "main subject or concept shown in the image. Consider educational content, "
                            "tutorials, documentaries, or explanatory videos."
                            + user_hint
                            + "\n\nRespond with ONLY the search query, nothing else."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=50,
    )
    search_query = vision_response.choices[0].message.content.strip().strip('"')

    pipeline = factory.build_pipeline(max_videos, provider)

    def stream_and_store():
        yield _json.dumps({"type": "query", "message": search_query}) + "\n"
        for chunk in pipeline.stream(search_query):
            try:
                parsed = _json.loads(chunk.strip())
                if parsed.get("type") == "video":
                    vid = parsed["data"]
                    transcript_store[vid["video_id"]] = vid.pop("transcript_text", "")
            except Exception:
                pass
            yield chunk

    return StreamingResponse(stream_and_store(), media_type="application/x-ndjson")


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
