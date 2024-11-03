"""
Tests for the images route.

This module contains tests for the image search endpoint, including
successful searches and error cases.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from main import app
from routes.images import router as images_router
from routes.shared.enums import OptimizationMode
from routes.shared.requests import ImageSearchRequest
from routes.shared.responses import StreamResponse

app.include_router(images_router)

client = TestClient(app)


def create_image_request(
    query: str = "test query",
    history: list = None,
    optimization_mode: OptimizationMode = OptimizationMode.BALANCED
) -> dict:
    """Create an image search request dictionary for testing."""
    return ImageSearchRequest(
        query=query,
        history=history or [],
        optimization_mode=optimization_mode,
        chat_model=None,
        embedding_model=None
    ).dict()


def create_mock_image_response() -> dict:
    """Create a mock image search response for testing."""
    return {
        "type": "response",
        "data": {
            "url": "http://example.com/image.jpg",
            "title": "Test Image",
            "source": "http://example.com"
        }
    }


@pytest.mark.asyncio
async def test_image_search_valid_request():
    """Test image search endpoint with valid request."""
    mock_response = create_mock_image_response()

    with patch('agents.image_search_agent.handle_image_search', new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = [mock_response]

        response = client.post(
            "/images/search",
            json=create_image_request()
        )
        
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 1
        assert result[0]["type"] == mock_response["type"]


@pytest.mark.asyncio
async def test_image_search_empty_query():
    """Test image search endpoint with empty query."""
    response = client.post(
        "/images/search",
        json=create_image_request(query="")
    )
    
    assert response.status_code == 400
    assert "Query cannot be empty" in response.json()["detail"]


@pytest.mark.asyncio
async def test_image_search_with_history():
    """Test image search endpoint with chat history."""
    mock_response = create_mock_image_response()
    history = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"}
    ]

    with patch('agents.image_search_agent.handle_image_search', new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = [mock_response]

        response = client.post(
            "/images/search",
            json=create_image_request(history=history)
        )
        
        assert response.status_code == 200
        # Verify history was passed to handler
        args = mock_handler.call_args
        assert len(args[0][1]) == 2  # Check history length


@pytest.mark.asyncio
async def test_image_search_server_error():
    """Test image search endpoint server error handling."""
    with patch('routes.images.setup_llm_and_embeddings', new_callable=AsyncMock) as mock_setup:
        mock_setup.side_effect = Exception("Server error")

        response = client.post(
            "/images/search",
            json=create_image_request()
        )
        
        assert response.status_code == 500
        assert "internal server error" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_image_search_with_optimization():
    """Test image search endpoint with different optimization modes."""
    mock_response = create_mock_image_response()

    with patch('agents.image_search_agent.handle_image_search', new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = [mock_response]

        response = client.post(
            "/images/search",
            json=create_image_request(
                optimization_mode=OptimizationMode.SPEED
            )
        )
        
        assert response.status_code == 200
        # Verify optimization mode was passed correctly
        args = mock_handler.call_args
        assert args[0][4] == "speed"  # Check optimization mode
