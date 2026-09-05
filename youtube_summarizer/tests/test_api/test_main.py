from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from api.main import app

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



