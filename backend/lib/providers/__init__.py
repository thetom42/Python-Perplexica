from .openai import load_openai_chat_models, load_openai_embeddings_models
from .anthropic import load_anthropic_chat_models
from .groq import load_groq_chat_models
from .ollama import load_ollama_chat_models, load_ollama_embeddings_models
from .transformers import load_transformers_embeddings_models
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from ...config import load_config

chat_model_providers = {
    "openai": load_openai_chat_models,
    "anthropic": load_anthropic_chat_models,
    "groq": load_groq_chat_models,
    "ollama": load_ollama_chat_models,
}

embedding_model_providers = {
    "openai": load_openai_embeddings_models,
    "local": load_transformers_embeddings_models,
    "ollama": load_ollama_embeddings_models,
}

async def get_available_chat_model_providers():
    models = {}
    for provider, loader in chat_model_providers.items():
        provider_models = await loader()
        if provider_models:
            models[provider] = provider_models
    models["custom_openai"] = {}
    return models

async def get_available_embedding_model_providers():
    models = {}
    for provider, loader in embedding_model_providers.items():
        provider_models = await loader()
        if provider_models:
            models[provider] = provider_models
    return models

def get_chat_model() -> BaseChatModel:
    config = load_config()
    provider = config.get("CHAT_MODEL", {}).get("PROVIDER", "openai")
    model_name = config.get("CHAT_MODEL", {}).get("MODEL", "gpt-3.5-turbo")
    
    if provider == "custom_openai":
        from langchain.chat_models import ChatOpenAI
        return ChatOpenAI(
            model_name=model_name,
            openai_api_key=config.get("CHAT_MODEL", {}).get("API_KEY"),
            base_url=config.get("CHAT_MODEL", {}).get("BASE_URL"),
        )
    
    models = chat_model_providers[provider]()
    return models.get(model_name)

def get_embeddings_model() -> Embeddings:
    config = load_config()
    provider = config.get("EMBEDDING_MODEL", {}).get("PROVIDER", "openai")
    model_name = config.get("EMBEDDING_MODEL", {}).get("MODEL", "text-embedding-ada-002")
    
    models = embedding_model_providers[provider]()
    return models.get(model_name)