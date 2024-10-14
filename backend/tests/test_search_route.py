import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from backend.main import app
from backend.routes.search import router as search_router

app.include_router(search_router, prefix="/api")

client = TestClient(app)

@pytest.mark.asyncio
async def test_web_search():
    mock_result = {
        "type": "response",
        "data": "This is a mock search result."
    }

    with patch('backend.routes.search.handle_web_search', new_callable=AsyncMock) as mock_handle_web_search:
        mock_handle_web_search.return_value = mock_result

        response = client.post("/api/search/web", json={
            "query": "test query",
            "history": []
        })

        assert response.status_code == 200
        assert response.json() == mock_result
        mock_handle_web_search.assert_called_once_with("test query", [], mock_handle_web_search.return_value, mock_handle_web_search.return_value)

@pytest.mark.asyncio
async def test_web_search_invalid_input():
    response = client.post("/api/search/web", json={
        "query": "",  # Empty query
        "history": []
    })

    assert response.status_code == 422  # Unprocessable Entity

@pytest.mark.asyncio
async def test_web_search_server_error():
    with patch('backend.routes.search.handle_web_search', new_callable=AsyncMock) as mock_handle_web_search:
        mock_handle_web_search.side_effect = Exception("Server error")

        response = client.post("/api/search/web", json={
            "query": "test query",
            "history": []
        })

        assert response.status_code == 500
        assert "error" in response.json()

# Add more tests for other search routes (academic, image, etc.) as needed
