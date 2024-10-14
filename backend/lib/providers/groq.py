from langchain_groq.chat_models import ChatGroq
from backend.config import get_groq_api_key

async def load_groq_chat_models():
    api_key = get_groq_api_key()
    if not api_key:
        return {}

    models = {
        "llama2-70b-4096": ChatGroq(model="llama2-70b-4096",
                                    temperature=0.0,
                                    max_retries=2,
                                    api_key=api_key),
        "mixtral-8x7b-32768": ChatGroq(model="mixtral-8x7b-32768",
                                       temperature=0.0,
                                       max_retries=2,
                                       api_key=api_key),
    }
    return models
