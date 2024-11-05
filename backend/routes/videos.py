"""
Video search route handlers.

This module provides endpoints for searching videos based on queries and chat history.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain.schema import HumanMessage, AIMessage
from agents.video_search_agent import handle_video_search
from providers import get_available_chat_model_providers
from utils.logger import logger


class VideoSearchRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]]
    chat_model_provider: Optional[str] = None
    chat_model: Optional[str] = None


class VideoSearchResponse(BaseModel):
    videos: List[Dict[str, Any]]


router = APIRouter(
    tags=["videos"],
    responses={
        500: {"description": "Internal server error"},
    }
)


@router.post(
    "/",
    response_model=VideoSearchResponse,
    description="Search for videos based on query",
    responses={
        200: {
            "description": "List of video search results",
            "model": VideoSearchResponse
        }
    }
)
async def video_search(request: VideoSearchRequest) -> VideoSearchResponse:
    """
    Search for videos based on the given query.
    
    Args:
        request: VideoSearchRequest containing search parameters
        
    Returns:
        VideoSearchResponse containing video results
        
    Raises:
        HTTPException: If an error occurs during video search
    """
    try:
        # Convert chat history to LangChain message types
        chat_history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user"
            else AIMessage(content=msg["content"])
            for msg in request.chat_history
        ]

        # Get available models and select default if not specified
        chat_models = await get_available_chat_model_providers()
        provider = request.chat_model_provider or next(iter(chat_models))
        model_name = request.chat_model or next(iter(chat_models[provider]))

        # Get the LLM model
        llm = None
        if provider in chat_models and model_name in chat_models[provider]:
            llm = chat_models[provider][model_name].model

        if not llm:
            raise HTTPException(
                status_code=500,
                detail="Invalid LLM model selected"
            )

        # Perform video search
        videos = await handle_video_search(
            {"chat_history": chat_history, "query": request.query},
            llm
        )

        return VideoSearchResponse(videos=videos)
    except Exception as e:
        logger.error("Error in video search: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail="An error has occurred."
        )
