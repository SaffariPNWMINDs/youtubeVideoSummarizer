"""
CLI entry point for Phase 0.

Run with:
    python -m youtube_summarizer.cli.main --query "machine learning"
    python -m youtube_summarizer.cli.main -q "machine learning" -n 5
"""

import argparse
import logging
import sys

from youtube_summarizer import factory


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

    print("\n📋  OVERVIEW\n")
    print(f"  {result.final_summary}\n")

    print("🔑  KEY TAKEAWAYS\n")
    for i, takeaway in enumerate(result.key_takeaways, 1):
        print(f"  {i:>2}. {takeaway}")

    print("\n📺  VIDEOS ANALYSED\n")
    for vs in result.video_summaries:
        print(f"  {vs.view_count:>12,} views  {vs.title[:55]}")
        print(f"  {' ' * 14}  {vs.video_url}")
        for point in vs.key_points:
            print(f"  {'':14}  • {point}")
        print()

    print(divider)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the top YouTube videos for any topic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python -m youtube_summarizer.cli.main -q 'LLM fine-tuning'",
    )
    parser.add_argument("--provider", "-p", choices=["claude", "openai"], default="claude", help="LLM provider to use (default: OpenAI's gpt-5)")
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
        pipeline = factory.build_pipeline(args.max_videos, args.provider)
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
