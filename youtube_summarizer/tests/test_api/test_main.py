import json

from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from api.main import app, _relay_pipeline_stream, transcript_store

client = TestClient(app)

def test_summarize_return_200(mocker):
    #create a fake result
    fake_result = MagicMock()
    fake_result.query = "machine learning"
    fake_result.final_summary = "Fake Summary"
    fake_result.key_takeaways = ["Takeaway 1", "Takeaway 2"]
    fake_result.video_summaries = []
    
    #patch the pipeline to return the fake result
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = fake_result
    mocker.patch('api.main.factory.build_pipeline', return_value=mock_pipeline)
    
    #call the API
    response = client.post("/summarize", json={"query": "machine learning", "max_videos": 5, "provider": "openai"})
    
    #assert the response
    assert response.status_code == 200
    assert response.json()["query"] == "machine learning"
    
def test_summarize_return_500_when_pipeline_fails(mocker):
    #patch the pipeline to raise an exception
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = None  # Simulate pipeline failure by returning None
    mocker.patch('api.main.factory.build_pipeline', return_value=mock_pipeline)
    
    #call the API
    response = client.post("/summarize", json={"query": "machine learning", "max_videos": 5, "provider": "openai"})
    
    #assert the response
    assert response.status_code == 500
    
def test_summarize_required_query(mocker):
    
    #call the API without query
    response = client.post("/summarize", json={"max_videos": 5, "provider": "openai"})
    
    #assert the response
    assert response.status_code == 422  # Unprocessable Entity due to missing required field


class TestRelayPipelineStream:
    """
    Regression tests for the bug behind the "Bad Unicode escape" /
    "Unterminated string in JSON" frontend errors: the relay used to
    yield the ORIGINAL chunk (with transcript_text still inside) even
    after popping transcript_text off the parsed copy, so the huge
    transcript payload — much larger for non-English text, since
    json.dumps escapes every non-ASCII char as \\uXXXX — was still sent
    to the client on every "video" event.
    """

    def test_strips_transcript_text_from_video_events(self):
        video_event = json.dumps({
            "type": "video",
            "data": {"video_id": "v1", "title": "Test", "transcript_text": "سلام دنیا " * 1000},
        }) + "\n"

        relayed = list(_relay_pipeline_stream([video_event]))

        assert len(relayed) == 1
        parsed = json.loads(relayed[0])
        assert "transcript_text" not in parsed["data"]
        assert transcript_store["v1"] == "سلام دنیا " * 1000

    def test_passes_through_non_video_events_unchanged(self):
        status_event = json.dumps({"type": "status", "message": "Searching..."}) + "\n"

        relayed = list(_relay_pipeline_stream([status_event]))

        assert relayed == [status_event]

    def test_passes_through_malformed_chunks_unchanged(self):
        broken_chunk = "not valid json\n"

        relayed = list(_relay_pipeline_stream([broken_chunk]))

        assert relayed == [broken_chunk]



