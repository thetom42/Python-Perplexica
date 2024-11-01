import pytest
from fastapi.testclient import TestClient
from main import app
from lib.providers import get_chat_model, get_embeddings_model
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.mark.asyncio
async def test_web_search_integration():
    mock_result = {
        "type": "response",
        "data": "This is a mock search result."
    }

    with patch('backend.agents.web_search_agent.handle_web_search', new_callable=AsyncMock) as mock_handle_web_search:
        mock_handle_web_search.return_value = mock_result

        response = client.post("/api/search/web", json={
            "query": "test query",
            "history": []
        })

        assert response.status_code == 200
        assert response.json() == mock_result
        mock_handle_web_search.assert_called_once()

@pytest.mark.asyncio
async def test_web_search_error_handling():
    with patch('backend.agents.web_search_agent.handle_web_search', new_callable=AsyncMock) as mock_handle_web_search:
        mock_handle_web_search.side_effect = Exception("Test error")

        response = client.post("/api/search/web", json={
            "query": "test query",
            "history": []
        })

        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()

@pytest.mark.asyncio
async def test_web_search_invalid_input():
    response = client.post("/api/search/web", json={
        "query": "",  # Empty query
        "history": []
    })

    assert response.status_code == 422  # Unprocessable Entity

# Add more integration tests for other endpoints as needed
