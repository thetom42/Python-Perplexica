"""
Image search route handlers.

This module provides endpoints for performing image searches using various
language models and processing the results.
"""

from typing import List, Dict, Any, Optional, cast, TypedDict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from providers import get_available_chat_model_providers
from agents.image_search_agent import handle_image_search
from utils.logger import logger


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str
    content: str


class ImageSearchRequest(BaseModel):
    """Image search request model."""
    query: str
    chat_history: List[ChatMessage]
    chat_model_provider: Optional[str] = None
    chat_model: Optional[str] = None


class ImageResult(BaseModel):
    """Image result model."""
    img_src: str
    url: str
    title: str


class ImageSearchResponse(BaseModel):
    """Image search response model."""
    images: List[ImageResult]


class ModelInfo(TypedDict):
    """Model information structure."""
    displayName: str
    model: BaseChatModel


router = APIRouter(
    tags=["images"],
    responses={
        500: {"description": "Internal server error"},
    }
)


@router.post(
    "/",
    response_model=ImageSearchResponse,
    description="Search for images based on the given query",
    responses={
        200: {
            "description": "Image search results",
            "model": ImageSearchResponse
        }
    }
)
async def search_images(request: ImageSearchRequest) -> ImageSearchResponse:
    """
    Search for images based on the given query and chat history.
    
    Args:
        request: The image search request containing query and chat history
        
    Returns:
        ImageSearchResponse containing the search results
        
    Raises:
        HTTPException: If an error occurs during the search
    """
    try:
        # Convert chat history to LangChain message types
        chat_history: List[BaseMessage] = [
            HumanMessage(content=msg.content) if msg.role == 'user'
            else AIMessage(content=msg.content)
            for msg in request.chat_history
        ]

        # Get available chat models and select the appropriate one
        chat_models = await get_available_chat_model_providers()
        provider = request.chat_model_provider or next(iter(chat_models))
        model_name = request.chat_model or next(iter(chat_models[provider]))

        # Validate model selection
        if not chat_models.get(provider) or not chat_models[provider].get(model_name):
            raise HTTPException(
                status_code=500,
                detail="Invalid LLM model selected"
            )

        # Get the model and ensure it's a BaseChatModel
        model_info = cast(ModelInfo, chat_models[provider][model_name])
        llm = model_info['model']

        # Create a dummy embeddings instance since it's not used for image search
        class DummyEmbeddings(Embeddings):
            async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
                return [[0.0]]

            async def aembed_query(self, text: str) -> List[float]:
                return [0.0]

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [[0.0]]

            def embed_query(self, text: str) -> List[float]:
                return [0.0]

        # Perform image search
        images = []
        async for result in handle_image_search(
            query=request.query,
            history=chat_history,
            llm=llm,
            embeddings=DummyEmbeddings(),
            optimization_mode="balanced"
        ):
            if result["type"] == "error":
                raise HTTPException(
                    status_code=500,
                    detail=result["data"]
                )
            elif result["type"] == "images":
                images = result["data"]
                break

        return ImageSearchResponse(images=images)

    except HTTPException:
        raise
    except Exception as err:
        logger.error("Error in image search: %s", str(err))
        raise HTTPException(
            status_code=500,
            detail="An error has occurred."
        ) from err
