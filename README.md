# YouTube Summarizer

[![CI](https://github.com/SaffariPNWMINDs/youtubeVideoSummarizer/actions/workflows/ci.yml/badge.svg)](https://github.com/SaffariPNWMINDs/youtubeVideoSummarizer/actions/workflows/ci.yml)

> Type a topic → the app searches YouTube, pulls transcripts from the top results, and returns an LLM-generated synthesis: per-video summaries, timestamped key points, a cross-video "what do these videos agree/disagree on" overview, and a chat box to ask follow-up questions grounded in each transcript.

A full-stack (FastAPI + React) application built to demonstrate practical, production-shaped patterns for working with LLMs — multi-provider abstraction, streaming responses, prompt design for structured output, and retrieval-augmented Q&A — on top of a cleanly layered, dependency-injected backend.

## Why this project

Most "LLM demo" projects are a single script that calls an API and prints a string. This one is built the way a real product would be: services behind interfaces, a pipeline that orchestrates them, results streamed to the client as they're produced, and a second LLM-powered subsystem (RAG) bolted on independently of the first. It's meant to show both **LLM application design** and **software engineering fundamentals** in one codebase.

## Core features

**Search & summarize**
- Searches YouTube (Data API v3) for a topic and ranks candidates by views, recency, or relevance
- Fetches each video's transcript and produces a structured per-video summary: 3–5 key points, a short prose summary, and a topic/percentage content breakdown
- Synthesizes all per-video summaries into one cross-video report — a consensus overview plus 5–7 ranked key takeaways — instead of just concatenating individual summaries
- **Analyze URLs mode** — paste specific YouTube links directly and skip the search step entirely
- Advanced filters: published-date range, duration (short/medium/long), minimum view count, channel-name filter, keyword exclusion, and result language

**LLM engineering**
- **Multi-provider abstraction** — swap between OpenAI (`gpt-4o-mini` / `gpt-4o`) and Anthropic Claude (`claude-haiku-4-5` / `claude-sonnet-4-6`) per request, behind a shared `BaseSummarizerService` interface
- **Two-stage "map-reduce" summarization** — a fast/cheap model summarizes each video independently (map), then a more capable model synthesizes all of them into one report (reduce), keeping cost proportional to depth rather than breadth
- **Structured output via prompting** — every LLM call is prompted for a strict JSON contract (key points, timestamps, category percentages) and parsed defensively (markdown-fence stripping, schema validation via Pydantic) rather than trusting free-form text
- **Streaming everywhere** — results stream to the client over NDJSON/SSE as each video finishes, and the RAG answer streams token-by-token, so the UI never blocks on the slowest step

**Retrieval-Augmented Generation (RAG)**
- Ask free-form questions about any analyzed video (`/ask`) — the transcript is chunked (with overlap), embedded (`text-embedding-3-small`), and stored in a per-video ChromaDB collection on first question
- Top-k semantic retrieval pulls the most relevant chunks for the question, which are injected into the prompt so answers are grounded in what the video actually said, not the model's general knowledge
- Answers auto-match the question's language (e.g. ask in Spanish or Farsi, get the answer back in the same language)

**Multimodal search (backend implemented)**
- Search by image — upload a photo, GPT-4o vision describes it into a search query
- Search by voice — record a clip, Whisper transcribes it, GPT-4o-mini turns it into a search query
- Search by short video — extracted frames are interpreted by GPT-4o to infer what's being shown and generate a query
- Google OAuth login that reads a user's YouTube subscriptions/liked videos and generates a personalized LLM greeting

*(These four are working API endpoints not yet wired into the current UI — next up on the roadmap below.)*

**Frontend**
- React + Vite SPA with incremental rendering as streamed results arrive
- Inline YouTube player that jumps straight to a key point's timestamp
- Per-video Q&A chat panel with streamed answers
- Content-breakdown bar charts per video

## Architecture

```
Query ──▶ SearchPipeline
            ├─ BaseVideoSearchService   (YouTube Data API v3)
            ├─ BaseTranscriptService    (youtube-transcript-api)
            └─ BaseSummarizerService    (OpenAI | Claude — pluggable)
                    │
                    ▼
            per-video summary (map) ──▶ cross-video synthesis (reduce)
```

The pipeline depends only on abstract base classes (`services/base.py`), never concrete implementations — classic Dependency Inversion / Strategy pattern. That's what makes it possible to:
- swap OpenAI ↔ Claude with one constructor argument
- test the entire pipeline with mock services and zero real API calls (see `tests/`)
- add new capabilities (RAG, multimodal search) as new modules without touching the core pipeline

Domain objects (`Video`, `Transcript`, `VideoSummary`, `AggregatedSummary`) are immutable, validated Pydantic models, not dicts — so a malformed LLM response fails fast at the parsing boundary instead of silently corrupting downstream state.

## Tech stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI, Uvicorn, streaming (NDJSON/SSE) |
| LLMs | OpenAI (`gpt-4o`, `gpt-4o-mini`, Whisper, embeddings), Anthropic Claude |
| RAG | ChromaDB (vector store), OpenAI embeddings, chunking with overlap |
| Data sources | YouTube Data API v3, `youtube-transcript-api`, Google OAuth |
| Validation | Pydantic / pydantic-settings |
| Frontend | React 19, Vite, rc-slider |
| Testing / CI | pytest, pytest-mock, pytest-cov, ruff, GitHub Actions |
| Packaging | Poetry |

## Quick start

### Backend

```bash
cd youtube_summarizer
poetry install
cp .env.example .env   # add YOUTUBE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
poetry run uvicorn api.main:app --reload
```

Or use the CLI directly, no server needed:

```bash
poetry run yt-summarize --query "machine learning basics" -n 5 --verbose
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the API at `http://localhost:8000` (see CORS config in `api/main.py`).

### Tests

```bash
cd youtube_summarizer
poetry run pytest -v
```

All tests run against mock services — no API keys or network calls required. CI runs `ruff check` and `pytest` on every push/PR to `main`.

## Project structure

```
youtube_summarizer/
├── api/main.py                      # FastAPI app — all HTTP endpoints
├── src/youtube_summarizer/
│   ├── config.py                    # Settings, loaded from .env
│   ├── factory.py                   # Wires real service implementations together
│   ├── models/                      # Pydantic domain models (Video, Transcript, Summary...)
│   ├── services/
│   │   ├── base.py                  # Abstract interfaces (the OOP contracts)
│   │   ├── youtube_search_service.py
│   │   ├── transcript_service.py
│   │   ├── openai_summarizer_service.py
│   │   └── summarizer_service.py    # Claude implementation
│   ├── pipeline/
│   │   ├── search_pipeline.py       # Orchestrates search → transcript → summarize → aggregate
│   │   └── rag_pipeline.py          # Chunk → embed → store → retrieve, for /ask
│   └── cli/main.py                  # CLI entry point
└── tests/                           # Mock-based unit tests
frontend/
└── src/App.jsx                      # React SPA
```

## Roadmap

| Stage | Status | Focus |
|---|---|---|
| CLI pipeline | ✅ Done | Search → transcript → summarize, OOP fundamentals |
| Web app (FastAPI + React) | ✅ Done | Streaming API, multi-provider LLMs, filters |
| RAG Q&A | ✅ Done | ChromaDB, embeddings, per-video chat |
| Multimodal search | 🔨 Backend done, UI pending | Image / voice / video-frame search entry points |
| Personalization | 🔨 Backend done, UI pending | Google OAuth, subscriptions-aware greeting |
| Caching & deployment | ⏳ Planned | Redis, Docker, CI/CD to a cloud host |
| Production hardening | ⏳ Planned | Persistent auth, observability, rate limiting |
