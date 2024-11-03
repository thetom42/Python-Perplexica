"""
Tests for the videos route.

This module contains tests for the video search endpoint, including successful
searches and error cases.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from main import app
from routes.videos import router as videos_router
from routes.shared.enums import OptimizationMode
from routes.shared.requests import VideoSearchRequest
from routes.shared.responses import VideoSearchResponse, VideoResult

app.include_router(videos_router)

client = TestClient(app)


def create_video_request(
    query: str = "test query",
    chat_history: list = None,
    optimization_mode: OptimizationMode = OptimizationMode.BALANCED
) -> dict:
    """Create a video search request dictionary for testing."""
    return VideoSearchRequest(
        query=query,
        chat_history=chat_history or [],
        optimization_mode=optimization_mode,
        chat_model=None,
        embedding_model=None
    ).dict()


def create_mock_video() -> dict:
    """Create a mock video result for testing."""
    return {
        "title": "Test Video",
        "url": "http://example.com/video",
        "thumbnail": "http://example.com/thumb.jpg",
        "description": "Test description",
        "duration": "10:00",
        "channel": "Test Channel"
    }


@pytest.mark.asyncio
async def test_video_search_valid_request():
    """Test video search endpoint with valid request."""
    mock_videos = [create_mock_video()]

    with patch('agents.video_search_agent.handle_video_search', new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = mock_videos

        response = client.post(
            "/videos",
            json=create_video_request()
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "videos" in result
        assert len(result["videos"]) == 1
        assert result["videos"][0]["title"] == "Test Video"


@pytest.mark.asyncio
async def test_video_search_empty_query():
    """Test video search endpoint with empty query."""
    response = client.post(
        "/videos",
        json=create_video_request(query="")
    )
    
    assert response.status_code == 400
    assert "Query cannot be empty" in response.json()["detail"]


@pytest.mark.asyncio
async def test_video_search_with_history():
    """Test video search endpoint with chat history."""
    mock_videos = [create_mock_video()]
    history = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"}
    ]

    with patch('agents.video_search_agent.handle_video_search', new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = mock_videos

        response = client.post(
            "/videos",
            json=create_video_request(chat_history=history)
        )
        
        assert response.status_code == 200
        # Verify history was passed to handler
        args = mock_handler.call_args
        assert len(args[1]["history"]) == 2


@pytest.mark.asyncio
async def test_video_search_server_error():
    """Test video search endpoint server error handling."""
    with patch('routes.videos.setup_llm_and_embeddings', new_callable=AsyncMock) as mock_setup:
        mock_setup.side_effect = Exception("Server error")

        response = client.post(
            "/videos",
            json=create_video_request()
        )
        
        assert response.status_code == 500
        assert "internal server error" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_video_search_with_optimization():
    """Test video search endpoint with different optimization modes."""
    mock_videos = [create_mock_video()]

    with patch('agents.video_search_agent.handle_video_search', new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = mock_videos

        response = client.post(
            "/videos",
            json=create_video_request(
                optimization_mode=OptimizationMode.SPEED
            )
        )
        
        assert response.status_code == 200
        # Verify optimization mode was passed correctly
        args = mock_handler.call_args
        assert args[1]["optimization_mode"] == "speed"
