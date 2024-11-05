"""
Model providers route handlers.

This module provides endpoints for retrieving available language and embedding
model providers and their configurations.
"""

from typing import Union, Dict, Any, TypedDict, cast
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings

from providers import (
    get_available_chat_model_providers,
    get_available_embedding_model_providers
)
from utils.logger import logger


class ModelProviderDict(TypedDict):
    """Model provider dictionary structure."""
    displayName: str


class ModelInfo(TypedDict):
    """Model information structure."""
    displayName: str
    model: Union[BaseChatModel, Embeddings]


class ModelProvidersResponse(BaseModel):
    """Response model for model providers endpoint."""
    chatModelProviders: Dict[str, Dict[str, ModelProviderDict]]
    embeddingModelProviders: Dict[str, Dict[str, ModelProviderDict]]


router = APIRouter(
    tags=["models"],
    responses={
        500: {"description": "Internal server error"},
    }
)


@router.get(
    "/",
    response_model=ModelProvidersResponse,
    description="Get available model providers",
    responses={
        200: {
            "description": "Available model providers",
            "model": ModelProvidersResponse
        }
    }
)
async def get_model_providers() -> ModelProvidersResponse:
    """
    Get available chat and embedding model providers.
    
    Returns:
        ModelProvidersResponse containing available providers and their models
        
    Raises:
        HTTPException: If providers cannot be retrieved
    """
    try:
        # Get both provider types concurrently
        chat_providers = await get_available_chat_model_providers()
        embedding_providers = await get_available_embedding_model_providers()

        # Remove model instances from chat providers
        chat_providers_filtered: Dict[str, Dict[str, ModelProviderDict]] = {}
        for provider, base_chat_models in chat_providers.items():
            chat_providers_filtered[provider] = {}
            for model_name, base_model_info in base_chat_models.items():
                if isinstance(base_model_info, dict) and 'displayName' in base_model_info:
                    chat_providers_filtered[provider][model_name] = {
                        'displayName': base_model_info['displayName']
                    }

        # Remove model instances from embedding providers
        embedding_providers_filtered: Dict[str, Dict[str, ModelProviderDict]] = {}
        for provider, embedding_models in embedding_providers.items():
            embedding_providers_filtered[provider] = {}
            for model_name, embedding_model_info in embedding_models.items():
                if isinstance(embedding_model_info, dict) and 'displayName' in embedding_model_info:
                    embedding_providers_filtered[provider][model_name] = {
                        'displayName': embedding_model_info['displayName']
                    }

        return ModelProvidersResponse(
            chatModelProviders=chat_providers_filtered,
            embeddingModelProviders=embedding_providers_filtered
        )
    except Exception as err:
        logger.error("Error getting model providers: %s", str(err))
        raise HTTPException(
            status_code=500,
            detail="An error has occurred."
        ) from err
