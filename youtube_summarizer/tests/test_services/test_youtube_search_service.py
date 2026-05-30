"""
Unit tests for YouTubeSearchService.

KEY LESSON — why we mock:
  Real API calls are slow, cost quota, and make tests flaky (network down?
  API key expired?). By mocking the Google client, we test OUR logic (parsing,
  sorting, error handling) without depending on YouTube's infrastructure.
"""

import pytest
from unittest.mock import MagicMock, patch

from youtube_summarizer.services.youtube_search_service import YouTubeSearchService
from youtube_summarizer.models.video import Video


# ── Fixtures ────────────────────────────────────────────────────────────────

MOCK_SEARCH_RESPONSE = {
    "items": [
        {"id": {"videoId": "abc123"}},
        {"id": {"videoId": "def456"}},
    ]
}

MOCK_VIDEOS_RESPONSE = {
    "items": [
        {
            "id": "abc123",
            "snippet": {
                "title": "Intro to Machine Learning",
                "channelTitle": "ML Academy",
                "publishedAt": "2024-01-15T10:00:00Z",
                "description": "A beginner's guide.",
            },
            "statistics": {"viewCount": "500000"},
        },
        {
            "id": "def456",
            "snippet": {
                "title": "Advanced Neural Networks",
                "channelTitle": "Deep Dive",
                "publishedAt": "2024-03-20T14:00:00Z",
                "description": "Advanced techniques.",
            },
            "statistics": {"viewCount": "1200000"},
        },
    ]
}


@pytest.fixture
def mock_youtube_client():
    """Returns a fully mocked Google API client."""
    client = MagicMock()

    # Chain mocks to match the builder pattern: client.search().list().execute()
    (
        client.search.return_value
        .list.return_value
        .execute.return_value
    ) = MOCK_SEARCH_RESPONSE

    (
        client.videos.return_value
        .list.return_value
        .execute.return_value
    ) = MOCK_VIDEOS_RESPONSE

    return client


@pytest.fixture
def service(mock_youtube_client):
    """YouTubeSearchService with the real Google client replaced by a mock."""
    with patch("youtube_summarizer.services.youtube_search_service.build") as mock_build:
        mock_build.return_value = mock_youtube_client
        svc = YouTubeSearchService()
        svc._client = mock_youtube_client      # inject directly to be safe
        yield svc


# ── Tests ────────────────────────────────────────────────────────────────────

class TestYouTubeSearchService:

    def test_search_returns_video_list(self, service):
        videos = service.search("machine learning")
        assert len(videos) == 2
        assert all(isinstance(v, Video) for v in videos)

    def test_videos_sorted_by_view_count_descending(self, service):
        videos = service.search("machine learning")
        view_counts = [v.view_count for v in videos]
        assert view_counts == sorted(view_counts, reverse=True)

    def test_video_url_is_generated(self, service):
        videos = service.search("machine learning")
        for video in videos:
            assert video.url.startswith("https://www.youtube.com/watch?v=")
            assert video.video_id in video.url

    def test_empty_search_returns_empty_list(self, service):
        service._client.search.return_value.list.return_value.execute.return_value = {
            "items": []
        }
        videos = service.search("xyzzy_nonexistent_topic_12345")
        assert videos == []

    def test_malformed_item_is_skipped(self, service):
        """A bad item in the response should not crash the whole search."""
        service._client.videos.return_value.list.return_value.execute.return_value = {
            "items": [
                MOCK_VIDEOS_RESPONSE["items"][0],
                {"id": "broken", "snippet": {}, "statistics": {}},  # missing keys
            ]
        }
        # Should not raise — malformed item is logged and skipped
        videos = service.search("machine learning")
        assert len(videos) >= 1
