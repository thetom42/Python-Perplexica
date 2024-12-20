from langchain_anthropic import ChatAnthropic
from config import get_anthropic_api_key
from utils.logger import logger

async def load_anthropic_chat_models():
    api_key = get_anthropic_api_key()
    if not api_key:
        return {}

    try:
        chat_models = {
            'claude-3-5-sonnet-20240620': {
                'displayName': 'Claude 3.5 Sonnet',
                'model': ChatAnthropic(
                    api_key=api_key,
                    model_name='claude-3-5-sonnet-20240620',
                    temperature=0.7,
                    timeout=60,
                    stop=['\n'],
                ),
            },
            'claude-3-opus-20240229': {
                'displayName': 'Claude 3 Opus',
                'model': ChatAnthropic(
                    api_key=api_key,
                    model_name='claude-3-opus-20240229',
                    temperature=0.7,
                    timeout=60,
                    stop=['\n'],
                ),
            },
            'claude-3-sonnet-20240229': {
                'displayName': 'Claude 3 Sonnet',
                'model': ChatAnthropic(
                    api_key=api_key,
                    model_name='claude-3-sonnet-20240229',
                    temperature=0.7,
                    timeout=60,
                    stop=['\n'],
                ),
            },
            'claude-3-haiku-20240307': {
                'displayName': 'Claude 3 Haiku',
                'model': ChatAnthropic(
                    api_key=api_key,
                    model_name='claude-3-haiku-20240307',
                    temperature=0.7,
                    timeout=60,
                    stop=['\n'],
                ),
            },
        }

        return chat_models
    except Exception as err:
        logger.error("Error loading Anthropic models: %s", err)
        return {}
