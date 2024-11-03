"""
Tests for the search route.

This module contains tests for the search endpoint, including different search modes
and error cases.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from main import app
from routes.search import router as search_router
from routes.shared.enums import FocusMode, OptimizationMode
from routes.shared.requests import SearchRequest
from routes.shared.responses import StreamResponse

app.include_router(search_router)

client = TestClient(app)


def create_search_request(
    query: str = "test query",
    focus_mode: FocusMode = FocusMode.WEB,
    optimization_mode: OptimizationMode = OptimizationMode.BALANCED,
    history: list = None
) -> dict:
    """Create a search request dictionary for testing."""
    return SearchRequest(
        query=query,
        focus_mode=focus_mode,
        optimization_mode=optimization_mode,
        history=history or [],
        chat_model=None,
        embedding_model=None
    ).dict()


@pytest.mark.asyncio
async def test_search_valid_request():
    """Test search endpoint with valid request."""
    mock_response = StreamResponse(
        type="response",
        data="This is a mock search result."
    ).dict()

    with patch('websocket.message_handler.search_handlers.get', new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = AsyncMock()
        mock_handler.return_value.__aiter__.return_value = [mock_response]

        response = client.post(
            "/search",
            json=create_search_request()
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"
        
        # Test streaming response content
        content = next(response.iter_lines()).decode()
        assert "This is a mock search result" in content


@pytest.mark.asyncio
async def test_search_empty_query():
    """Test search endpoint with empty query."""
    response = client.post(
        "/search",
        json=create_search_request(query="")
    )
    
    assert response.status_code == 400
    assert "Query cannot be empty" in response.json()["detail"]


@pytest.mark.asyncio
async def test_search_with_history():
    """Test search endpoint with chat history."""
    history = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"}
    ]
    
    mock_response = StreamResponse(
        type="response",
        data="Response with context"
    ).dict()

    with patch('websocket.message_handler.search_handlers.get', new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = AsyncMock()
        mock_handler.return_value.__aiter__.return_value = [mock_response]

        response = client.post(
            "/search",
            json=create_search_request(history=history)
        )
        
        assert response.status_code == 200
        
        # Verify history was passed to handler
        args = mock_handler.return_value.call_args
        assert len(args[0][1]) == 2  # Check history length


@pytest.mark.asyncio
async def test_search_with_sources():
    """Test search endpoint with sources in response."""
    mock_responses = [
        StreamResponse(type="response", data="Search result").dict(),
        StreamResponse(type="sources", data=[{"url": "http://example.com"}]).dict()
    ]

    with patch('websocket.message_handler.search_handlers.get', new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = AsyncMock()
        mock_handler.return_value.__aiter__.return_value = mock_responses

        response = client.post(
            "/search",
            json=create_search_request()
        )
        
        assert response.status_code == 200
        
        # Test streaming response content
        content = [line.decode() for line in response.iter_lines()]
        assert len(content) == 2
        assert '"sources"' in content[1]


@pytest.mark.asyncio
async def test_search_server_error():
    """Test search endpoint server error handling."""
    with patch('routes.search.setup_llm_and_embeddings', new_callable=AsyncMock) as mock_setup:
        mock_setup.side_effect = Exception("Server error")

        response = client.post(
            "/search",
            json=create_search_request()
        )
        
        assert response.status_code == 500
        assert "internal server error" in response.json()["detail"].lower()
