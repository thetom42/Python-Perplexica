"""
Tests for the suggestions route.

This module contains tests for the suggestions endpoint, including successful
suggestion generation and error cases.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from main import app
from routes.suggestions import router as suggestions_router
from routes.shared.enums import OptimizationMode
from routes.shared.requests import BaseLLMRequest
from routes.shared.responses import SuggestionResponse

app.include_router(suggestions_router)

client = TestClient(app)


def create_suggestion_request(
    history: list = None,
    optimization_mode: OptimizationMode = OptimizationMode.BALANCED
) -> dict:
    """Create a suggestion request dictionary for testing."""
    return BaseLLMRequest(
        history=history or [],
        optimization_mode=optimization_mode
    ).dict()


@pytest.mark.asyncio
async def test_suggestions_valid_request():
    """Test suggestions endpoint with valid request."""
    mock_suggestions = ["suggestion1", "suggestion2"]

    with patch('agents.suggestion_generator_agent.handle_suggestion_generation', new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = mock_suggestions

        response = client.post(
            "/suggestions",
            json=create_suggestion_request(
                history=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"}
                ]
            )
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "suggestions" in result
        assert len(result["suggestions"]) == 2
        assert result["suggestions"] == mock_suggestions


@pytest.mark.asyncio
async def test_suggestions_empty_history():
    """Test suggestions endpoint with empty chat history."""
    mock_suggestions = []

    with patch('agents.suggestion_generator_agent.handle_suggestion_generation', new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = mock_suggestions
        
        response = client.post(
            "/suggestions",
            json=create_suggestion_request()
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "suggestions" in result
        assert len(result["suggestions"]) == 0


@pytest.mark.asyncio
async def test_suggestions_server_error():
    """Test suggestions endpoint server error handling."""
    with patch('routes.suggestions.setup_llm_and_embeddings', new_callable=AsyncMock) as mock_setup:
        mock_setup.side_effect = Exception("Server error")

        response = client.post(
            "/suggestions",
            json=create_suggestion_request(
                history=[{"role": "user", "content": "Hello"}]
            )
        )
        
        assert response.status_code == 500
        assert "internal server error" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_suggestions_with_optimization_mode():
    """Test suggestions endpoint with different optimization modes."""
    mock_suggestions = ["optimized suggestion"]

    with patch('agents.suggestion_generator_agent.handle_suggestion_generation', new_callable=AsyncMock) as mock_handler:
        mock_handler.return_value = mock_suggestions

        response = client.post(
            "/suggestions",
            json=create_suggestion_request(
                optimization_mode=OptimizationMode.SPEED
            )
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["suggestions"] == mock_suggestions
        
        # Verify optimization mode was passed correctly
        call_args = mock_handler.call_args
        assert call_args.kwargs["optimization_mode"] == "speed"
