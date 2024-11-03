"""
Configuration route handlers.

This module provides endpoints for managing application configuration,
including model settings and API keys.
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from providers import (
    get_available_chat_model_providers,
    get_available_embedding_model_providers
)
from config import (
    get_openai_api_key,
    get_ollama_api_endpoint,
    get_anthropic_api_key,
    get_groq_api_key,
    update_config
)
from utils.logger import logger
from .shared.responses import ModelsResponse
from .shared.config import APIConfig, ModelInfo
from .shared.exceptions import ServerError

router = APIRouter(
    prefix="/config",
    tags=["config"],
    responses={
        500: {"description": "Internal server error"},
    }
)


async def get_chat_model() -> BaseChatModel:
    """
    Get the default chat model.
    
    Returns:
        BaseChatModel: Default chat model instance
        
    Raises:
        ServerError: If chat model cannot be loaded
    """
    try:
        chat_models: Dict[str, Dict[str, BaseChatModel]] = await get_available_chat_model_providers()
        provider = next(iter(chat_models))
        model_name = next(iter(chat_models[provider]))
        return chat_models[provider][model_name]
    except Exception as e:
        logger.error("Error getting chat model: %s", str(e))
        raise ServerError("Failed to load chat model") from e


async def get_embeddings_model() -> Embeddings:
    """
    Get the default embeddings model.
    
    Returns:
        Embeddings: Default embeddings model instance
        
    Raises:
        ServerError: If embeddings model cannot be loaded
    """
    try:
        embedding_models = await get_available_embedding_model_providers()
        provider = next(iter(embedding_models))
        model_name = next(iter(embedding_models[provider]))
        return embedding_models[provider][model_name]
    except Exception as e:
        logger.error("Error getting embeddings model: %s", str(e))
        raise ServerError("Failed to load embeddings model") from e


@router.get(
    "/",
    response_model=ModelsResponse,
    description="Get current configuration",
    responses={
        200: {
            "description": "Current configuration",
            "model": ModelsResponse
        }
    }
)
async def get_config() -> ModelsResponse:
    """
    Get current configuration including available models and API keys.
    
    Returns:
        ModelsResponse containing current configuration
        
    Raises:
        ServerError: If configuration cannot be retrieved
    """
    try:
        chat_model_providers = await get_available_chat_model_providers()
        embedding_model_providers = await get_available_embedding_model_providers()

        return ModelsResponse(
            chat_model_providers={
                provider: [
                    ModelInfo(name=model, display_name=info.display_name)
                    for model, info in models.items()
                ]
                for provider, models in chat_model_providers.items()
            },
            embedding_model_providers={
                provider: [
                    ModelInfo(name=model, display_name=info.display_name)
                    for model, info in models.items()
                ]
                for provider, models in embedding_model_providers.items()
            }
        )
    except Exception as e:
        logger.error("Error getting config: %s", str(e))
        raise ServerError() from e


@router.post(
    "/",
    response_model=DeleteResponse,
    description="Update configuration",
    responses={
        200: {
            "description": "Configuration updated successfully",
            "model": DeleteResponse
        }
    }
)
async def update_config_route(config: APIConfig) -> DeleteResponse:
    """
    Update configuration with new API keys and endpoints.
    
    Args:
        config: New configuration values
        
    Returns:
        DeleteResponse confirming update
        
    Raises:
        ServerError: If configuration update fails
    """
    try:
        new_config = {
            "API_KEYS": {
                "OPENAI": config.openai_api_key,
                "GROQ": config.groq_api_key,
                "ANTHROPIC": config.anthropic_api_key,
            },
            "API_ENDPOINTS": {
                "OLLAMA": config.ollama_api_url,
            }
        }
        update_config(new_config)
        return DeleteResponse(message="Config updated successfully")
    except Exception as e:
        logger.error("Error updating config: %s", str(e))
        raise ServerError() from e
