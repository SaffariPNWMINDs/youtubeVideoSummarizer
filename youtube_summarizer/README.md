# YouTube Summarizer

> Search any topic → fetch top YouTube videos → extract transcripts → LLM-powered summary.

This is **Phase 0** of a progressively built ML app. See the [development roadmap](#roadmap) below.

---

## Quick Start

### Prerequisites
- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation) (`pip install poetry`)
- A [YouTube Data API v3](https://console.cloud.google.com) key
- An [Anthropic API](https://console.anthropic.com) key

### Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/YOUR_USERNAME/youtube-summarizer.git
cd youtube-summarizer

# 2. Install dependencies
poetry install

# 3. Copy and fill in your API keys
cp .env.example .env
# Edit .env with your actual keys

# 4. Run
poetry run yt-summarize --query "machine learning basics"

# Or with more options:
poetry run yt-summarize -q "LLM fine-tuning" -n 5 --save --verbose
```

### Example output

```
────────────────────────────────────────────────────
  TOPIC: MACHINE LEARNING BASICS
  8 videos analysed
────────────────────────────────────────────────────

📋  OVERVIEW

  Machine learning is a branch of AI that enables computers to learn from data
  without explicit programming. Across the analysed videos, experts agree that
  supervised learning is the best entry point for beginners...

🔑  KEY TAKEAWAYS

   1. Start with supervised learning — it has the clearest mental model
   2. Understand bias-variance tradeoff before choosing model complexity
   ...

📺  VIDEOS ANALYSED

    1,200,000 views  Machine Learning for Beginners — Full Course
                     https://youtube.com/watch?v=...
                     • ML is about pattern recognition from examples
                     • Three main types: supervised, unsupervised, reinforcement
                     • scikit-learn is the best library to start with
```

---

## Running Tests

```bash
poetry run pytest -v
```

Tests use mock services — no real API calls needed.

---

## Project Structure

```
youtube_summarizer/
├── src/youtube_summarizer/
│   ├── config.py                   # Settings (loaded from .env)
│   ├── models/
│   │   ├── video.py                # Video domain model
│   │   ├── transcript.py           # Transcript domain model
│   │   └── summary.py              # VideoSummary + AggregatedSummary models
│   ├── services/
│   │   ├── base.py                 # Abstract base classes (the OOP contracts)
│   │   ├── youtube_search_service.py
│   │   ├── transcript_service.py
│   │   └── summarizer_service.py
│   ├── pipeline/
│   │   └── search_pipeline.py      # Orchestrates the full flow
│   └── cli/
│       └── main.py                 # Entry point
└── tests/
    ├── test_services/
    └── test_pipeline/
```

### Key OOP patterns used

| Pattern | Where | Why |
|---|---|---|
| Abstract Base Class | `services/base.py` | Defines contracts, enables DI |
| Strategy | All services | Swap implementations without changing the pipeline |
| Dependency Injection | `SearchPipeline.__init__` | Testable, decoupled |
| Pydantic models | `models/` | Validated, immutable domain objects |
| Factory function | `cli/main.py: build_pipeline()` | Centralised wiring of real dependencies |

---

## Roadmap

| Version | Goal | Key tech |
|---|---|---|
| **Phase 0** ✅ | CLI working locally | Python, Pydantic, LangChain |
| **V0** | Basic web app deployed | FastAPI, React, Docker, AWS EC2 |
| **V1** | Caching, streaming, CI/CD | Redis, ElastiCache, RDS, GitHub Actions |
| **V2** | RAG + Q&A on transcripts | ChromaDB, embeddings, Langfuse |
| **V3** | Production-ready | ECS/Fargate, Cognito, CloudWatch |
| **V4** | Monetization | Stripe, CloudFront |
