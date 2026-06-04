
from youtube_summarizer.pipeline.search_pipeline import SearchPipeline
from youtube_summarizer.services.summarizer_service import ClaudeSummarizerService
from youtube_summarizer.services.openai_summarizer_service import OpenAISummarizerService
from youtube_summarizer.services.transcript_service import YouTubeTranscriptService
from youtube_summarizer.services.youtube_search_service import YouTubeSearchService



def build_pipeline(max_videos: int, provider: str) -> SearchPipeline:
    """
    Factory function — builds the pipeline with real service implementations.
    In tests, you'd call SearchPipeline(...) directly with mock services.
    """
    if provider == "openai":
        summarizer_service = OpenAISummarizerService()
    else:
        summarizer_service = ClaudeSummarizerService()

    return SearchPipeline(
        search_service=YouTubeSearchService(),
        transcript_service=YouTubeTranscriptService(),
        summarizer_service=summarizer_service,
        max_videos=max_videos,
    )