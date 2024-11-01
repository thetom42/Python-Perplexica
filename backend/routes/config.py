from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from lib.providers import (
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

router = APIRouter()

class ConfigUpdate(BaseModel):
    openaiApiKey: str
    ollamaApiUrl: str
    anthropicApiKey: str
    groqApiKey: str

async def get_chat_model() -> BaseChatModel:
    """Get the default chat model."""
    try:
        chat_models = await get_available_chat_model_providers()
        provider = next(iter(chat_models))
        model_name = next(iter(chat_models[provider]))
        return chat_models[provider][model_name].model
    except Exception as e:
        logger.error("Error getting chat model: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to load chat model") from e

async def get_embeddings_model() -> Embeddings:
    """Get the default embeddings model."""
    try:
        embedding_models = await get_available_embedding_model_providers()
        provider = next(iter(embedding_models))
        model_name = next(iter(embedding_models[provider]))
        return embedding_models[provider][model_name].model
    except Exception as e:
        logger.error("Error getting embeddings model: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to load embeddings model") from e

@router.get("/")
async def get_config() -> Dict[str, Any]:
    try:
        config = {}
        chat_model_providers = await get_available_chat_model_providers()
        embedding_model_providers = await get_available_embedding_model_providers()
        config['chatModelProviders'] = {}
        config['embeddingModelProviders'] = {}
        for provider, models in chat_model_providers.items():
            config['chatModelProviders'][provider] = [
                {"name": model, "displayName": info.displayName}
                for model, info in models.items()
            ]
        for provider, models in embedding_model_providers.items():
            config['embeddingModelProviders'][provider] = [
                {"name": model, "displayName": info.displayName}
                for model, info in models.items()
            ]
        config['openaiApiKey'] = get_openai_api_key()
        config['ollamaApiUrl'] = get_ollama_api_endpoint()
        config['anthropicApiKey'] = get_anthropic_api_key()
        config['groqApiKey'] = get_groq_api_key()
        return config
    except Exception as e:
        logger.error("Error getting config: %s", str(e))
        raise HTTPException(status_code=500, detail="An error has occurred.") from e

@router.post("/")
async def update_config_route(config: ConfigUpdate) -> Dict[str, str]:
    try:
        new_config = {
            "API_KEYS": {
                "OPENAI": config.openaiApiKey,
                "GROQ": config.groqApiKey,
                "ANTHROPIC": config.anthropicApiKey,
            },
            "API_ENDPOINTS": {
                "OLLAMA": config.ollamaApiUrl,
            }
        }
        update_config(new_config)
        return {"message": "Config updated"}
    except Exception as e:
        logger.error("Error updating config: %s", str(e))
        raise HTTPException(status_code=500, detail="An error has occurred.") from e
