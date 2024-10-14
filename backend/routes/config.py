from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from backend.lib.providers import get_available_chat_model_providers, get_available_embedding_model_providers
from backend.config import get_openai_api_key, get_ollama_api_endpoint, get_anthropic_api_key, get_groq_api_key, update_config
from backend.utils.logger import logger

router = APIRouter()

class ConfigUpdate(BaseModel):
    openaiApiKey: str
    ollamaApiUrl: str
    anthropicApiKey: str
    groqApiKey: str

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
        logger.error(f"Error getting config: {str(e)}")
        raise HTTPException(status_code=500, detail="An error has occurred.")

@router.post("/")
async def update_config_route(config: ConfigUpdate) -> Dict[str, str]:
    try:
        updated_config = {
            "API_KEYS": {
                "OPENAI": config.openaiApiKey,
                "GROQ": config.groqApiKey,
                "ANTHROPIC": config.anthropicApiKey,
            },
            "API_ENDPOINTS": {
                "OLLAMA": config.ollamaApiUrl,
            },
        }

        update_config(updated_config)
        return {"message": "Config updated"}
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}")
        raise HTTPException(status_code=500, detail="An error has occurred.")
