"""
Tests for the images route.

This module contains tests for the image search endpoint, including
successful searches and error cases.
"""

import pytest
from typing import Dict, Any, List, Optional, Union
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from main import app
from routes.images import (
    router as images_router,
    ImageSearchRequest,
    ChatMessage
)

app.include_router(images_router)

client = TestClient(app)


def create_mock_chat_model() -> MagicMock:
    """Create a mock chat model for testing."""
    mock_model = MagicMock()
    mock_model.displayName = "Test Model"
    return mock_model


def create_image_request(
    query: str = "test query",
    chat_history: Optional[List[Dict[str, str]]] = None,
    chat_model_provider: Optional[str] = None,
    chat_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an image search request dictionary for testing.
    
    Args:
        query: The search query
        chat_history: Optional list of chat messages
        chat_model_provider: Optional model provider name
        chat_model: Optional model name
        
    Returns:
        Dict containing the request data
    """
    if chat_history is None:
        chat_history = []

    # Convert history to ChatMessage objects
    chat_messages = [
        ChatMessage(role=msg["role"], content=msg["content"])
        for msg in chat_history
    ]

    request = ImageSearchRequest(
        query=query,
        chat_history=chat_messages,
        chat_model_provider=chat_model_provider,
        chat_model=chat_model
    )
    return request.model_dump()


def create_mock_image_result() -> Dict[str, Any]:
    """Create a mock image search result for testing."""
    return {
        "type": "images",
        "data": [{
            "img_src": "http://example.com/image.jpg",
            "url": "http://example.com",
            "title": "Test Image"
        }]
    }


@pytest.mark.asyncio
async def test_image_search_valid_request():
    """Test image search endpoint with valid request."""
    mock_result = create_mock_image_result()
    mock_models = {
        "test_provider": {
            "test_model": {
                "displayName": "Test Model",
                "model": create_mock_chat_model()
            }
        }
    }

    with patch('routes.images.get_available_chat_model_providers', new_callable=AsyncMock) as mock_providers, \
         patch('routes.images.handle_image_search', new_callable=AsyncMock) as mock_handler:
        mock_providers.return_value = mock_models
        mock_handler.return_value = AsyncMock(__aiter__=lambda _: iter([mock_result]))

        response = client.post(
            "/images",
            json=create_image_request()
        )

        assert response.status_code == 200
        result = response.json()
        assert "images" in result
        assert len(result["images"]) == 1
        assert result["images"][0]["img_src"] == mock_result["data"][0]["img_src"]


@pytest.mark.asyncio
async def test_image_search_with_history():
    """Test image search endpoint with chat history."""
    mock_result = create_mock_image_result()
    mock_models = {
        "test_provider": {
            "test_model": {
                "displayName": "Test Model",
                "model": create_mock_chat_model()
            }
        }
    }
    chat_history = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"}
    ]

    with patch('routes.images.get_available_chat_model_providers', new_callable=AsyncMock) as mock_providers, \
         patch('routes.images.handle_image_search', new_callable=AsyncMock) as mock_handler:
        mock_providers.return_value = mock_models
        mock_handler.return_value = AsyncMock(__aiter__=lambda _: iter([mock_result]))

        response = client.post(
            "/images",
            json=create_image_request(chat_history=chat_history)
        )

        assert response.status_code == 200
        mock_handler.assert_called_once()
        # Verify history was passed to handler
        args = mock_handler.call_args
        assert len(args[1]["history"]) == 2  # Check history length


@pytest.mark.asyncio
async def test_image_search_with_specific_model():
    """Test image search endpoint with specific model selection."""
    mock_result = create_mock_image_result()
    mock_models = {
        "test_provider": {
            "test_model": {
                "displayName": "Test Model",
                "model": create_mock_chat_model()
            }
        }
    }

    with patch('routes.images.get_available_chat_model_providers', new_callable=AsyncMock) as mock_providers, \
         patch('routes.images.handle_image_search', new_callable=AsyncMock) as mock_handler:
        mock_providers.return_value = mock_models
        mock_handler.return_value = AsyncMock(__aiter__=lambda _: iter([mock_result]))

        response = client.post(
            "/images",
            json=create_image_request(
                chat_model_provider="test_provider",
                chat_model="test_model"
            )
        )

        assert response.status_code == 200
        mock_handler.assert_called_once()


@pytest.mark.asyncio
async def test_image_search_invalid_model():
    """Test image search endpoint with invalid model selection."""
    mock_models = {
        "test_provider": {
            "test_model": {
                "displayName": "Test Model",
                "model": create_mock_chat_model()
            }
        }
    }

    with patch('routes.images.get_available_chat_model_providers', new_callable=AsyncMock) as mock_providers:
        mock_providers.return_value = mock_models

        response = client.post(
            "/images",
            json=create_image_request(
                chat_model_provider="invalid_provider",
                chat_model="invalid_model"
            )
        )

        assert response.status_code == 500
        assert "Invalid LLM model selected" in response.json()["detail"]


@pytest.mark.asyncio
async def test_image_search_error_response():
    """Test image search endpoint error handling."""
    mock_error = {
        "type": "error",
        "data": "Test error message"
    }
    mock_models = {
        "test_provider": {
            "test_model": {
                "displayName": "Test Model",
                "model": create_mock_chat_model()
            }
        }
    }

    with patch('routes.images.get_available_chat_model_providers', new_callable=AsyncMock) as mock_providers, \
         patch('routes.images.handle_image_search', new_callable=AsyncMock) as mock_handler:
        mock_providers.return_value = mock_models
        mock_handler.return_value = AsyncMock(__aiter__=lambda _: iter([mock_error]))

        response = client.post(
            "/images",
            json=create_image_request()
        )

        assert response.status_code == 500
        assert mock_error["data"] in response.json()["detail"]


@pytest.mark.asyncio
async def test_image_search_server_error():
    """Test image search endpoint server error handling."""
    with patch('routes.images.get_available_chat_model_providers', new_callable=AsyncMock) as mock_providers:
        mock_providers.side_effect = Exception("Server error")

        response = client.post(
            "/images",
            json=create_image_request()
        )

        assert response.status_code == 500
        assert "An error has occurred" in response.json()["detail"]
