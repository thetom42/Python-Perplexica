from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.embeddings.base import OpenAIEmbeddings
from backend.config import get_openai_api_key

async def load_openai_chat_models():
    api_key = get_openai_api_key()
    if not api_key:
        return {}

    models = {
        "gpt-3.5-turbo": ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key),
        "gpt-4": ChatOpenAI(model="gpt-4", api_key=api_key),
    }
    return models

async def load_openai_embeddings_models():
    api_key = get_openai_api_key()
    if not api_key:
        return {}

    models = {
        "text-embedding-ada-002": OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key),
    }
    return models
