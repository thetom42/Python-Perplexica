from fastapi import APIRouter
from backend.lib.providers import get_available_chat_model_providers, get_available_embedding_model_providers

router = APIRouter()

@router.get("/chat")
async def get_chat_models():
    return await get_available_chat_model_providers()

@router.get("/embeddings")
async def get_embedding_models():
    return await get_available_embedding_model_providers()
