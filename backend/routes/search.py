"""
Search route handlers.

This module provides endpoints for performing searches with different models and modes.
"""

from typing import List, Optional, Dict, Any, Tuple, Union
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from providers import get_available_chat_model_providers, get_available_embedding_model_providers
from websocket.message_handler import search_handlers
from utils.logger import logger


class ChatModel(BaseModel):
    provider: str
    model: str
    customOpenAIBaseURL: Optional[str] = None
    customOpenAIKey: Optional[str] = None


class EmbeddingModel(BaseModel):
    provider: str
    model: str


class SearchRequest(BaseModel):
    optimizationMode: str = "balanced"
    focusMode: str
    chatModel: Optional[ChatModel] = None
    embeddingModel: Optional[EmbeddingModel] = None
    query: str
    history: List[Tuple[str, str]] = []


class SearchResponse(BaseModel):
    message: str
    sources: List[Dict[str, Any]]


router = APIRouter(
    tags=["search"],
    responses={
        500: {"description": "Internal server error"},
    }
)


@router.post(
    "/",
    response_model=SearchResponse,
    description="Perform a search based on query",
    responses={
        200: {
            "description": "Search results with sources",
            "model": SearchResponse
        }
    }
)
async def search(body: SearchRequest) -> SearchResponse:
    """
    Perform a search based on the specified parameters.
    
    Args:
        body: SearchRequest containing search parameters
        
    Returns:
        SearchResponse containing results and sources
        
    Raises:
        HTTPException: If an error occurs during search
    """
    try:
        if not body.focusMode or not body.query:
            raise HTTPException(
                status_code=400,
                detail="Missing focus mode or query"
            )

        # Convert history to LangChain messages
        history: List[BaseMessage] = [
            HumanMessage(content=msg[1]) if msg[0] == "human"
            else AIMessage(content=msg[1])
            for msg in body.history
        ]

        # Get available models
        chat_models = await get_available_chat_model_providers()
        embedding_models = await get_available_embedding_model_providers()

        # Get chat model
        chat_provider = body.chatModel.provider if body.chatModel else next(iter(chat_models))
        chat_model_name = body.chatModel.model if body.chatModel else next(iter(chat_models[chat_provider]))

        # Get embedding model
        embedding_provider = body.embeddingModel.provider if body.embeddingModel else next(iter(embedding_models))
        embedding_model_name = body.embeddingModel.model if body.embeddingModel else next(iter(embedding_models[embedding_provider]))

        # Setup LLM
        llm: Optional[BaseChatModel] = None
        if body.chatModel and body.chatModel.provider == "custom_openai":
            if not body.chatModel.customOpenAIBaseURL or not body.chatModel.customOpenAIKey:
                raise HTTPException(
                    status_code=400,
                    detail="Missing custom OpenAI base URL or key"
                )
            llm = ChatOpenAI(
                model_name=body.chatModel.model,
                openai_api_key=body.chatModel.customOpenAIKey,
                temperature=0.7,
                openai_api_base=body.chatModel.customOpenAIBaseURL
            )
        elif chat_provider in chat_models and chat_model_name in chat_models[chat_provider]:
            llm = chat_models[chat_provider][chat_model_name].model

        # Setup embeddings
        embeddings: Optional[Embeddings] = None
        if embedding_provider in embedding_models and embedding_model_name in embedding_models[embedding_provider]:
            embeddings = embedding_models[embedding_provider][embedding_model_name].model

        if not llm or not embeddings:
            raise HTTPException(
                status_code=400,
                detail="Invalid model selected"
            )

        # Get search handler
        search_handler = search_handlers.get(body.focusMode)
        if not search_handler:
            raise HTTPException(
                status_code=400,
                detail="Invalid focus mode"
            )

        # Perform search
        message = ""
        sources = []
        async for data in search_handler(
            body.query,
            history,
            llm,
            embeddings,
            body.optimizationMode
        ):
            parsed_data = data if isinstance(data, dict) else {}
            if parsed_data.get("type") == "response":
                message += parsed_data.get("data", "")
            elif parsed_data.get("type") == "sources":
                sources = parsed_data.get("data", [])

        return SearchResponse(message=message, sources=sources)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in search: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail="An error has occurred."
        )
