from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from backend.config import get_ollama_api_endpoint

async def load_ollama_chat_models():
    base_url = get_ollama_api_endpoint()
    if not base_url:
        return {}

    models = {
        "llama2": ChatOllama(model="llama2", base_url=base_url),
        "mistral": ChatOllama(model="mistral", base_url=base_url),
    }
    return models

async def load_ollama_embeddings_models():
    base_url = get_ollama_api_endpoint()
    if not base_url:
        return {}

    models = {
        "llama2": OllamaEmbeddings(model="llama2", base_url=base_url),
    }
    return models
