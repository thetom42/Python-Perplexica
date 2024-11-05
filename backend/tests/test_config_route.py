"""
Tests for the config route.

This module contains tests for the configuration endpoints, including
model settings and API key management.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from main import app
from routes.config import router as config_router, ModelDict

app.include_router(config_router)

client = TestClient(app)


def create_mock_model_info() -> dict:
    """Create mock model information for testing."""
    return {
        "name": "test-model",
        "displayName": "Test Model"
    }


def create_mock_api_config() -> dict:
    """Create mock API configuration for testing."""
    return {
        "openaiApiKey": "test-openai-key",
        "ollamaApiUrl": "http://test.ollama.api",
        "anthropicApiKey": "test-anthropic-key",
        "groqApiKey": "test-groq-key"
    }


@pytest.mark.asyncio
async def test_get_config():
    """Test getting current configuration."""
    mock_model: ModelDict = {
        "displayName": "Model 1",
        "model": MagicMock()
    }
    mock_providers = {
        "test_provider": {
            "model1": mock_model,
            "model2": {"displayName": "Model 2", "model": MagicMock()}
        }
    }

    with patch('routes.config.get_available_chat_model_providers', new_callable=AsyncMock) as mock_chat_providers, \
         patch('routes.config.get_available_embedding_model_providers', new_callable=AsyncMock) as mock_embed_providers, \
         patch('routes.config.get_openai_api_key', return_value="test-key"), \
         patch('routes.config.get_ollama_api_endpoint', return_value="test-url"), \
         patch('routes.config.get_anthropic_api_key', return_value="test-key"), \
         patch('routes.config.get_groq_api_key', return_value="test-key"):

        mock_chat_providers.return_value = mock_providers
        mock_embed_providers.return_value = mock_providers

        response = client.get("/config")

        assert response.status_code == 200
        result = response.json()
        assert "chatModelProviders" in result
        assert "embeddingModelProviders" in result
        assert len(result["chatModelProviders"]["test_provider"]) == 2
        assert result["chatModelProviders"]["test_provider"][0]["displayName"] == "Model 1"
        assert result["openaiApiKey"] == "test-key"
        assert result["ollamaApiUrl"] == "test-url"


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
        assert result["message"] == "Config updated"

        # Verify config was updated with correct values
        mock_update.assert_called_once()
        call_args = mock_update.call_args[0][0]
        assert call_args["API_KEYS"]["OPENAI"] == config_data["openaiApiKey"]
        assert call_args["API_ENDPOINTS"]["OLLAMA"] == config_data["ollamaApiUrl"]


@pytest.mark.asyncio
async def test_get_chat_model():
    """Test getting chat models."""
    mock_model: ModelDict = {
        "displayName": "Model 1",
        "model": MagicMock()
    }
    mock_providers = {
        "test_provider": {
            "model1": mock_model
        }
    }

    with patch('routes.config.get_available_chat_model_providers', new_callable=AsyncMock) as mock_providers_func, \
         patch('routes.config.get_available_embedding_model_providers', return_value={}), \
         patch('routes.config.get_openai_api_key', return_value=None), \
         patch('routes.config.get_ollama_api_endpoint', return_value=None), \
         patch('routes.config.get_anthropic_api_key', return_value=None), \
         patch('routes.config.get_groq_api_key', return_value=None):

        mock_providers_func.return_value = mock_providers

        response = client.get("/config")
        assert response.status_code == 200
        result = response.json()
        assert "chatModelProviders" in result
        assert result["chatModelProviders"]["test_provider"][0]["displayName"] == "Model 1"


@pytest.mark.asyncio
async def test_get_embeddings_model():
    """Test getting embeddings models."""
    mock_model: ModelDict = {
        "displayName": "Model 1",
        "model": MagicMock()
    }
    mock_providers = {
        "test_provider": {
            "model1": mock_model
        }
    }

    with patch('routes.config.get_available_embedding_model_providers', new_callable=AsyncMock) as mock_providers_func, \
         patch('routes.config.get_available_chat_model_providers', return_value={}), \
         patch('routes.config.get_openai_api_key', return_value=None), \
         patch('routes.config.get_ollama_api_endpoint', return_value=None), \
         patch('routes.config.get_anthropic_api_key', return_value=None), \
         patch('routes.config.get_groq_api_key', return_value=None):

        mock_providers_func.return_value = mock_providers

        response = client.get("/config")
        assert response.status_code == 200
        result = response.json()
        assert "embeddingModelProviders" in result
        assert result["embeddingModelProviders"]["test_provider"][0]["displayName"] == "Model 1"


@pytest.mark.asyncio
async def test_config_error_handling():
    """Test error handling in configuration endpoints."""
    with patch('routes.config.get_available_chat_model_providers', new_callable=AsyncMock) as mock_providers:
        mock_providers.side_effect = Exception("Provider error")

        response = client.get("/config")
        assert response.status_code == 500
        assert response.json()["detail"] == "An error has occurred."


@pytest.mark.asyncio
async def test_invalid_config_update():
    """Test updating configuration with invalid data."""
    invalid_config = {
        "openaiApiKey": "",  # Empty key
        "ollamaApiUrl": "invalid-url",
        "anthropicApiKey": "",
        "groqApiKey": ""
    }

    with patch('routes.config.update_config') as mock_update:
        response = client.post(
            "/config",
            json=invalid_config
        )

        assert response.status_code == 200  # Valid as empty strings are allowed
        mock_update.assert_called_once()  # Config update should be called
