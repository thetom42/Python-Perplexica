"""
Configuration route handlers.

This module provides endpoints for managing application configuration,
including model settings and API keys.
"""

from typing import Dict, Any, List, TypedDict, cast
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from providers import (
    get_available_chat_model_providers,
    get_available_embedding_model_providers
)
from config import (
    get_openai_api_key,
    get_ollama_api_endpoint,
    get_anthropic_api_key,
    get_groq_api_key,
    update_config,
    PartialConfig
)
from utils.logger import logger


class ModelInfo(BaseModel):
    name: str
    displayName: str


class ConfigResponse(BaseModel):
    chatModelProviders: Dict[str, List[ModelInfo]]
    embeddingModelProviders: Dict[str, List[ModelInfo]]
    openaiApiKey: str | None
    ollamaApiUrl: str | None
    anthropicApiKey: str | None
    groqApiKey: str | None


class ConfigUpdateRequest(BaseModel):
    openaiApiKey: str | None = None
    ollamaApiUrl: str | None = None
    anthropicApiKey: str | None = None
    groqApiKey: str | None = None


class MessageResponse(BaseModel):
    message: str


class ModelDict(TypedDict):
    displayName: str
    model: Any


router = APIRouter(
    tags=["config"],
    responses={
        500: {"description": "Internal server error"},
    }
)


@router.get(
    "/",
    response_model=ConfigResponse,
    description="Get current configuration",
    responses={
        200: {
            "description": "Current configuration",
            "model": ConfigResponse
        }
    }
)
async def get_config() -> ConfigResponse:
    """
    Get current configuration including available models and API keys.
    
    Returns:
        ConfigResponse containing current configuration
        
    Raises:
        HTTPException: If configuration cannot be retrieved
    """
    try:
        chat_model_providers = await get_available_chat_model_providers()
        embedding_model_providers = await get_available_embedding_model_providers()

        return ConfigResponse(
            chatModelProviders={
                provider: [
                    ModelInfo(
                        name=model_name,
                        displayName=cast(ModelDict, model_info)['displayName']
                    )
                    for model_name, model_info in models.items()
                ]
                for provider, models in chat_model_providers.items()
            },
            embeddingModelProviders={
                provider: [
                    ModelInfo(
                        name=model_name,
                        displayName=cast(ModelDict, model_info)['displayName']
                    )
                    for model_name, model_info in models.items()
                ]
                for provider, models in embedding_model_providers.items()
            },
            openaiApiKey=get_openai_api_key(),
            ollamaApiUrl=get_ollama_api_endpoint(),
            anthropicApiKey=get_anthropic_api_key(),
            groqApiKey=get_groq_api_key()
        )
    except Exception as err:
        logger.error("Error getting config: %s", str(err))
        raise HTTPException(status_code=500, detail="An error has occurred.") from err


@router.post(
    "/",
    response_model=MessageResponse,
    description="Update configuration",
    responses={
        200: {
            "description": "Configuration updated successfully",
            "model": MessageResponse
        }
    }
)
async def update_config_route(config: ConfigUpdateRequest) -> MessageResponse:
    """
    Update configuration with new API keys and endpoints.
    
    Args:
        config: New configuration values
        
    Returns:
        MessageResponse confirming update
        
    Raises:
        HTTPException: If configuration update fails
    """
    try:
        # Create a properly typed config dictionary
        api_keys: Dict[str, str | None] = {
            "OPENAI": config.openaiApiKey,
            "GROQ": config.groqApiKey,
            "ANTHROPIC": config.anthropicApiKey,
        }
        api_endpoints: Dict[str, str | None] = {
            "OLLAMA": config.ollamaApiUrl,
        }
        new_config = cast(PartialConfig, {
            "API_KEYS": api_keys,
            "API_ENDPOINTS": api_endpoints,
        })

        update_config(new_config)
        return MessageResponse(message="Config updated")
    except Exception as err:
        logger.error("Error updating config: %s", str(err))
        raise HTTPException(status_code=500, detail="An error has occurred.") from err
