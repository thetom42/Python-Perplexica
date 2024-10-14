from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.embeddings.base import OpenAIEmbeddings
from backend.config import get_openai_api_key
from backend.utils.logger import logger

async def load_openai_chat_models():
    api_key = get_openai_api_key()
    if not api_key:
        return {}

    try:
        chat_models = {
            'gpt-3.5-turbo': {
                'displayName': 'GPT-3.5 Turbo',
                'model': ChatOpenAI(
                    model='gpt-3.5-turbo',
                    api_key=api_key,
                    temperature=0.7,
                ),
            },
            'gpt-4': {
                'displayName': 'GPT-4',
                'model': ChatOpenAI(
                    model='gpt-4',
                    api_key=api_key,
                    temperature=0.7,
                ),
            },
            'gpt-4-turbo': {
                'displayName': 'GPT-4 turbo',
                'model': ChatOpenAI(
                    model='gpt-4-turbo',
                    api_key=api_key,
                    temperature=0.7,
                ),
            },
            'gpt-4o': {
                'displayName': 'GPT-4 omni',
                'model': ChatOpenAI(
                    model='gpt-4o',
                    api_key=api_key,
                    temperature=0.7,
                ),
            },
            'gpt-4o-mini': {
                'displayName': 'GPT-4 omni mini',
                'model': ChatOpenAI(
                    model='gpt-4o-mini',
                    api_key=api_key,
                    temperature=0.7,
                ),
            },
        }

        return chat_models
    except Exception as err:
        logger.error(f"Error loading OpenAI models: {err}")
        return {}

async def load_openai_embeddings_models():
    api_key = get_openai_api_key()
    if not api_key:
        return {}

    try:
        embedding_models = {
            'text-embedding-3-small': {
                'displayName': 'Text Embedding 3 Small',
                'model': OpenAIEmbeddings(
                    model='text-embedding-3-small',
                    api_key=api_key,
                ),
            },
            'text-embedding-3-large': {
                'displayName': 'Text Embedding 3 Large',
                'model': OpenAIEmbeddings(
                    model='text-embedding-3-large',
                    api_key=api_key,
                ),
            },
        }

        return embedding_models
    except Exception as err:
        logger.error(f"Error loading OpenAI embeddings model: {err}")
        return {}
