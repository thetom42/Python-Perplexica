from langchain_anthropic.chat_models import ChatAnthropic
from backend.config import get_anthropic_api_key

async def load_anthropic_chat_models():
    api_key = get_anthropic_api_key()
    if not api_key:
        return {}

    models = {
        "claude-2": ChatAnthropic(model="claude-3-sonnet-20240229",
                                  temperature=0,
                                  max_tokens=1024,
                                  timeout=None,
                                  max_retries=2,
                                  api_key=api_key),
        "claude-instant-1": ChatAnthropic(model="claude-instant-1", 
                                          temperature=0,
                                          max_tokens=1024,
                                          timeout=None,
                                          max_retries=2,
                                          api_key=api_key)
    }
    return models
