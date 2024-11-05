"""
Tests for the models route.

This module contains tests for the model providers endpoint, including
successful retrieval and error cases.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from main import app
from routes.models import router as models_router

app.include_router(models_router)

client = TestClient(app)


def create_mock_model() -> MagicMock:
    """Create a mock model for testing."""
    return MagicMock()


def create_mock_provider_response() -> dict:
    """Create a mock provider response for testing."""
    return {
        "test_provider": {
            "model1": {
                "displayName": "Test Model 1",
                "model": create_mock_model()
            },
            "model2": {
                "displayName": "Test Model 2",
                "model": create_mock_model()
            }
        }
    }


@pytest.mark.asyncio
async def test_get_model_providers():
    """Test getting available model providers."""
    mock_providers = create_mock_provider_response()

    with patch('routes.models.get_available_chat_model_providers', new_callable=AsyncMock) as mock_chat_providers, \
         patch('routes.models.get_available_embedding_model_providers', new_callable=AsyncMock) as mock_embed_providers:

        mock_chat_providers.return_value = mock_providers
        mock_embed_providers.return_value = mock_providers

        response = client.get("/models")

        assert response.status_code == 200
        result = response.json()

        # Check structure
        assert "chatModelProviders" in result
        assert "embeddingModelProviders" in result

        # Check chat providers
        chat_providers = result["chatModelProviders"]
        assert "test_provider" in chat_providers
        assert len(chat_providers["test_provider"]) == 2
        assert "model1" in chat_providers["test_provider"]
        assert chat_providers["test_provider"]["model1"]["displayName"] == "Test Model 1"

        # Verify model instances were removed
        assert "model" not in chat_providers["test_provider"]["model1"]

        # Check embedding providers
        embed_providers = result["embeddingModelProviders"]
        assert "test_provider" in embed_providers
        assert len(embed_providers["test_provider"]) == 2
        assert "model1" in embed_providers["test_provider"]
        assert embed_providers["test_provider"]["model1"]["displayName"] == "Test Model 1"

        # Verify model instances were removed
        assert "model" not in embed_providers["test_provider"]["model1"]


@pytest.mark.asyncio
async def test_get_model_providers_empty():
    """Test getting model providers when none are available."""
    with patch('routes.models.get_available_chat_model_providers', new_callable=AsyncMock) as mock_chat_providers, \
         patch('routes.models.get_available_embedding_model_providers', new_callable=AsyncMock) as mock_embed_providers:

        mock_chat_providers.return_value = {}
        mock_embed_providers.return_value = {}

        response = client.get("/models")

        assert response.status_code == 200
        result = response.json()
        assert "chatModelProviders" in result
        assert "embeddingModelProviders" in result
        assert len(result["chatModelProviders"]) == 0
        assert len(result["embeddingModelProviders"]) == 0


@pytest.mark.asyncio
async def test_get_model_providers_error():
    """Test error handling when getting model providers fails."""
    with patch('routes.models.get_available_chat_model_providers', new_callable=AsyncMock) as mock_providers:
        mock_providers.side_effect = Exception("Provider error")

        response = client.get("/models")

        assert response.status_code == 500
        assert "An error has occurred" in response.json()["detail"]


@pytest.mark.asyncio
async def test_model_provider_structure():
    """Test the structure of model provider responses."""
    mock_providers = {
        "test_provider": {
            "model1": {
                "displayName": "Test Model 1",
                "model": create_mock_model(),
                "extra_field": "should be removed"
            }
        }
    }

    with patch('routes.models.get_available_chat_model_providers', new_callable=AsyncMock) as mock_chat_providers, \
         patch('routes.models.get_available_embedding_model_providers', new_callable=AsyncMock) as mock_embed_providers:

        mock_chat_providers.return_value = mock_providers
        mock_embed_providers.return_value = mock_providers

        response = client.get("/models")

        assert response.status_code == 200
        result = response.json()

        # Verify only displayName is included
        provider = result["chatModelProviders"]["test_provider"]["model1"]
        assert list(provider.keys()) == ["displayName"]
        assert provider["displayName"] == "Test Model 1"
