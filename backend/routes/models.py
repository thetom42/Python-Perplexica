from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from backend.lib.providers import get_available_chat_model_providers, get_available_embedding_model_providers
from backend.utils.logger import logger

router = APIRouter()

@router.get("/")
async def get_models() -> Dict[str, Any]:
    try:
        chat_model_providers = await get_available_chat_model_providers()
        embedding_model_providers = await get_available_embedding_model_providers()

        # Remove the 'model' field from each provider's model information
        for provider in chat_model_providers.values():
            for model_info in provider.values():
                model_info.pop('model', None)

        for provider in embedding_model_providers.values():
            for model_info in provider.values():
                model_info.pop('model', None)

        return {
            "chatModelProviders": chat_model_providers,
            "embeddingModelProviders": embedding_model_providers
        }
    except Exception as e:
        logger.error("Error in getting models: %s", str(e))
        raise HTTPException(status_code=500, detail="An error has occurred.") from e
