import aiohttp
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from backend.config import get_ollama_api_endpoint
from backend.utils.logger import logger

async def load_ollama_chat_models():
    base_url = get_ollama_api_endpoint()
    if not base_url:
        return {}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/tags") as response:
                data = await response.json()
                ollama_models = data.get("models", [])

        chat_models = {}
        for model in ollama_models:
            chat_models[model["model"]] = {
                "displayName": model["name"],
                "model": ChatOllama(
                    base_url=base_url,
                    model=model["model"],
                    temperature=0.7,
                ),
            }

        return chat_models
    except Exception as err:
        logger.error("Error loading Ollama models: %s", err)
        return {}

async def load_ollama_embeddings_models():
    base_url = get_ollama_api_endpoint()
    if not base_url:
        return {}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/tags") as response:
                data = await response.json()
                ollama_models = data.get("models", [])

        embeddings_models = {}
        for model in ollama_models:
            embeddings_models[model["model"]] = {
                "displayName": model["name"],
                "model": OllamaEmbeddings(
                    base_url=base_url,
                    model=model["model"],
                ),
            }

        return embeddings_models
    except Exception as err:
        logger.error("Error loading Ollama embeddings models: %s", err)
        return {}
