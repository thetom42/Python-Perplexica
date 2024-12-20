from langchain_openai import ChatOpenAI
from config import get_groq_api_key
from utils.logger import logger

async def load_groq_chat_models():
    api_key = get_groq_api_key()
    if not api_key:
        return {}

    try:
        chat_models = {
            'llama-3.2-3b-preview': {
                'displayName': 'Llama 3.2 3B',
                'model': ChatOpenAI(
                    api_key=api_key,
                    name='llama-3.2-3b-preview',
                    temperature=0.7,
                    base_url='https://api.groq.com/openai/v1'
                ),
            },
            'llama-3.2-11b-text-preview': {
                'displayName': 'Llama 3.2 11B Text',
                'model': ChatOpenAI(
                    api_key=api_key,
                    name='llama-3.2-11b-text-preview',
                    temperature=0.7,
                    base_url='https://api.groq.com/openai/v1'
                ),
            },
            'llama-3.2-90b-text-preview': {
                'displayName': 'Llama 3.2 90B Text',
                'model': ChatOpenAI(
                    api_key=api_key,
                    name='llama-3.2-90b-text-preview',
                    temperature=0.7,
                    base_url='https://api.groq.com/openai/v1'
                ),
            },
            'llama-3.1-70b-versatile': {
                'displayName': 'Llama 3.1 70B',
                'model': ChatOpenAI(
                    api_key=api_key,
                    name='llama-3.1-70b-versatile',
                    temperature=0.7,
                    base_url='https://api.groq.com/openai/v1'
                ),
            },
            'llama-3.1-8b-instant': {
                'displayName': 'Llama 3.1 8B',
                'model': ChatOpenAI(
                    api_key=api_key,
                    name='llama-3.1-8b-instant',
                    temperature=0.7,
                    base_url='https://api.groq.com/openai/v1'
                ),
            },
            'llama3-8b-8192': {
                'displayName': 'LLama3 8B',
                'model': ChatOpenAI(
                    api_key=api_key,
                    name='llama3-8b-8192',
                    temperature=0.7,
                    base_url='https://api.groq.com/openai/v1'
                ),
            },
            'llama3-70b-8192': {
                'displayName': 'LLama3 70B',
                'model': ChatOpenAI(
                    api_key=api_key,
                    name='llama3-70b-8192',
                    temperature=0.7,
                    base_url='https://api.groq.com/openai/v1'
                ),
            },
            'mixtral-8x7b-32768': {
                'displayName': 'Mixtral 8x7B',
                'model': ChatOpenAI(
                    api_key=api_key,
                    name='mixtral-8x7b-32768',
                    temperature=0.7,
                    base_url='https://api.groq.com/openai/v1'
                ),
            },
            'gemma-7b-it': {
                'displayName': 'Gemma 7B',
                'model': ChatOpenAI(
                    api_key=api_key,
                    name='gemma-7b-it',
                    temperature=0.7,
                    base_url='https://api.groq.com/openai/v1'
                ),
            },
            'gemma2-9b-it': {
                'displayName': 'Gemma2 9B',
                'model': ChatOpenAI(
                    api_key=api_key,
                    name='gemma2-9b-it',
                    temperature=0.7,
                    base_url='https://api.groq.com/openai/v1'
                ),
            },
        }

        return chat_models
    except Exception as err:
        logger.error("Error loading Groq models: %s", err)
        return {}
