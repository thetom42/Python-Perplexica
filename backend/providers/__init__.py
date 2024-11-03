from typing import Dict, Optional, Any
from providers.openai import load_openai_chat_models, load_openai_embeddings_models
from providers.anthropic import load_anthropic_chat_models
from providers.groq import load_groq_chat_models
from providers.ollama import load_ollama_chat_models, load_ollama_embeddings_models
from providers.transformers import load_transformers_embeddings_models
from langchain_core.utils.utils import convert_to_secret_str
from langchain.chat_models.base import BaseChatModel
from langchain_openai.chat_models.base import ChatOpenAI
from langchain.embeddings.base import Embeddings
import config

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

async def get_available_chat_model_providers() -> Dict[str, Dict[str, BaseChatModel]]:
    models: Dict[str, Dict[str, BaseChatModel]] = {}
    for provider, loader in chat_model_providers.items():
        provider_models = await loader()
        if provider_models:
            models[provider] = provider_models
    models["custom_openai"] = {}
    return models

async def get_available_embedding_model_providers() -> Dict[str, Dict[str, Embeddings]]:
    models: Dict[str, Dict[str, Embeddings]] = {}
    for provider, loader in embedding_model_providers.items():
        provider_models = await loader()
        if provider_models:
            models[provider] = provider_models
    return models

def get_chat_model() -> Any:
    chat_config = config.get_chat_model()
    provider = chat_config["PROVIDER"]
    model_name = chat_config["NAME"]

    if provider == "custom_openai":
        # For custom OpenAI-compatible endpoints, we need to get the API key and base URL
        # from the environment variables directly since they're not part of the standard config
        return ChatOpenAI(
            name=model_name,
            api_key=convert_to_secret_str(config.get_openai_api_key()),
            base_url=config.get_api_endpoints().get("OPENAI", ""),
        )

    models = chat_model_providers[provider]()
    return models.get(model_name)

def get_embeddings_model() -> Any:
    embedding_config = config.get_embedding_model()
    provider = embedding_config["PROVIDER"]
    model_name = embedding_config["NAME"]

    models = embedding_model_providers[provider]()
    return models.get(model_name)
