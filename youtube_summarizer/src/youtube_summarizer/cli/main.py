"""
CLI entry point for Phase 0.

Run with:
    python -m youtube_summarizer.cli.main --query "machine learning"
    python -m youtube_summarizer.cli.main -q "machine learning" -n 5
"""

import argparse
import logging
import sys

from youtube_summarizer.pipeline.search_pipeline import SearchPipeline
from youtube_summarizer.services.summarizer_service import ClaudeSummarizerService
from youtube_summarizer.services.transcript_service import YouTubeTranscriptService
from youtube_summarizer.services.youtube_search_service import YouTubeSearchService


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy third-party loggers
    logging.getLogger("googleapiclient").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def print_result(result) -> None:
    divider = "─" * 60

    print(f"\n{divider}")
    print(f"  TOPIC: {result.query.upper()}")
    print(f"  {result.video_count} videos analysed")
    print(divider)

    print(f"\n📋  OVERVIEW\n")
    print(f"  {result.final_summary}\n")

    print(f"🔑  KEY TAKEAWAYS\n")
    for i, takeaway in enumerate(result.key_takeaways, 1):
        print(f"  {i:>2}. {takeaway}")

    print(f"\n📺  VIDEOS ANALYSED\n")
    for vs in result.video_summaries:
        print(f"  {vs.view_count:>12,} views  {vs.title[:55]}")
        print(f"  {' ' * 14}  {vs.video_url}")
        for point in vs.key_points:
            print(f"  {'':14}  • {point}")
        print()

    print(divider)
    print()


def build_pipeline(max_videos: int) -> SearchPipeline:
    """
    Factory function — builds the pipeline with real service implementations.
    In tests, you'd call SearchPipeline(...) directly with mock services.
    """
    return SearchPipeline(
        search_service=YouTubeSearchService(),
        transcript_service=YouTubeTranscriptService(),
        summarizer_service=ClaudeSummarizerService(),
        max_videos=max_videos,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the top YouTube videos for any topic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python -m youtube_summarizer.cli.main -q 'LLM fine-tuning'",
    )
    parser.add_argument("--query", "-q", required=True, help="Topic to search")
    parser.add_argument(
        "--max-videos", "-n", type=int, default=10,
        help="Max videos to fetch (default: 10)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    parser.add_argument(
        "--save", "-s", action="store_true",
        help="Save summary as markdown file"
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        pipeline = build_pipeline(args.max_videos)
        result = pipeline.run(args.query)

        if result is None:
            print("\n❌  Could not generate a summary. Check logs for details.")
            sys.exit(1)

        print_result(result)

        if args.save:
            filename = f"summary_{result.query.replace(' ', '_')}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result.to_markdown())
            print(f"💾  Saved to {filename}\n")

    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
