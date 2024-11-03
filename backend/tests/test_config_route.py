"""
Tests for the config route.

This module contains tests for the configuration endpoints, including
model settings and API key management.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from main import app
from routes.config import router as config_router
from routes.shared.config import APIConfig, ModelInfo

app.include_router(config_router)

client = TestClient(app)


def create_mock_model_info() -> dict:
    """Create mock model information for testing."""
    return {
        "name": "test-model",
        "display_name": "Test Model"
    }


def create_mock_api_config() -> dict:
    """Create mock API configuration for testing."""
    return {
        "openai_api_key": "test-openai-key",
        "ollama_api_url": "http://test.ollama.api",
        "anthropic_api_key": "test-anthropic-key",
        "groq_api_key": "test-groq-key"
    }


@pytest.mark.asyncio
async def test_get_config():
    """Test getting current configuration."""
    mock_providers = {
        "test_provider": {
            "model1": MagicMock(display_name="Model 1"),
            "model2": MagicMock(display_name="Model 2")
        }
    }

    with patch('routes.config.get_available_chat_model_providers', new_callable=AsyncMock) as mock_chat_providers, \
         patch('routes.config.get_available_embedding_model_providers', new_callable=AsyncMock) as mock_embed_providers:
        
        mock_chat_providers.return_value = mock_providers
        mock_embed_providers.return_value = mock_providers

        response = client.get("/config")
        
        assert response.status_code == 200
        result = response.json()
        assert "chat_model_providers" in result
        assert "embedding_model_providers" in result
        assert len(result["chat_model_providers"]["test_provider"]) == 2


@pytest.mark.asyncio
async def test_update_config():
    """Test updating configuration."""
    config_data = create_mock_api_config()

    with patch('routes.config.update_config') as mock_update:
        response = client.post(
            "/config",
            json=config_data
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "message" in result
        assert "successfully" in result["message"].lower()
        
        # Verify config was updated with correct values
        mock_update.assert_called_once()
        call_args = mock_update.call_args[0][0]
        assert call_args["API_KEYS"]["OPENAI"] == config_data["openai_api_key"]
        assert call_args["API_ENDPOINTS"]["OLLAMA"] == config_data["ollama_api_url"]


@pytest.mark.asyncio
async def test_get_chat_model():
    """Test getting default chat model."""
    mock_providers = {
        "test_provider": {
            "model1": MagicMock(display_name="Model 1")
        }
    }

    with patch('routes.config.get_available_chat_model_providers', new_callable=AsyncMock) as mock_providers_func:
        mock_providers_func.return_value = mock_providers
        
        response = client.get("/config")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_embeddings_model():
    """Test getting default embeddings model."""
    mock_providers = {
        "test_provider": {
            "model1": MagicMock(display_name="Model 1")
        }
    }

    with patch('routes.config.get_available_embedding_model_providers', new_callable=AsyncMock) as mock_providers_func:
        mock_providers_func.return_value = mock_providers
        
        response = client.get("/config")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_config_error_handling():
    """Test error handling in configuration endpoints."""
    with patch('routes.config.get_available_chat_model_providers', new_callable=AsyncMock) as mock_providers:
        mock_providers.side_effect = Exception("Provider error")
        
        response = client.get("/config")
        assert response.status_code == 500
        assert "internal server error" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_invalid_config_update():
    """Test updating configuration with invalid data."""
    invalid_config = {
        "openai_api_key": "",  # Empty key
        "ollama_api_url": "invalid-url",
        "anthropic_api_key": "",
        "groq_api_key": ""
    }

    response = client.post(
        "/config",
        json=invalid_config
    )
    
    assert response.status_code == 422  # Validation error
