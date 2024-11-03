from typing import List, Tuple, Optional, Any
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain.chat_models.openai import ChatOpenAI
from fastapi import HTTPException
from routes.shared.config import ChatModel, EmbeddingModel
from routes.shared.requests import ChatMessage
from providers import get_available_chat_model_providers, get_available_embedding_model_providers


def convert_chat_history(messages: List[ChatMessage]) -> List[BaseMessage]:
    """Convert chat messages from API format to LangChain format."""
    return [
        HumanMessage(content=msg.content) if msg.role == "user"
        else AIMessage(content=msg.content)
        for msg in messages
    ]


async def setup_llm_and_embeddings(
    chat_model: Optional[ChatModel],
    embedding_model: Optional[EmbeddingModel] = None
) -> Tuple[BaseChatModel, Embeddings]:
    """Set up LLM and embeddings models based on configuration."""
    chat_model_providers = await get_available_chat_model_providers()
    embedding_model_providers = await get_available_embedding_model_providers()

    # Setup LLM
    llm: Optional[BaseChatModel] = None
    if chat_model and chat_model.provider == "custom_openai":
        if not chat_model.custom_openai_base_url or not chat_model.custom_openai_key:
            raise HTTPException(
                status_code=400,
                detail="Missing custom OpenAI base URL or key"
            )
        llm = ChatOpenAI(
            model=chat_model.model,
            api_key=chat_model.custom_openai_key,
            temperature=0.7,
            base_url=chat_model.custom_openai_base_url,
        )
    else:
        provider = (chat_model.provider if chat_model
                   else next(iter(chat_model_providers)))
        model = (chat_model.model if chat_model
                else next(iter(chat_model_providers[provider])))

        if (provider in chat_model_providers and
            model in chat_model_providers[provider]):
            llm = chat_model_providers[provider][model]

    # Setup embeddings
    embeddings: Optional[Embeddings] = None
    if embedding_model:
        provider = embedding_model.provider
        model = embedding_model.model
        if (provider in embedding_model_providers and
            model in embedding_model_providers[provider]):
            embeddings = embedding_model_providers[provider][model]
    else:
        # Get default embedding model
        provider = next(iter(embedding_model_providers))
        model = next(iter(embedding_model_providers[provider]))
        embeddings = embedding_model_providers[provider][model]

    if not llm or not embeddings:
        raise HTTPException(
            status_code=400,
            detail="Invalid model configuration"
        )

    return llm, embeddings
