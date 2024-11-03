"""
Video search route handlers.

This module provides endpoints for searching videos based on queries and chat history.
"""

from fastapi import APIRouter, Depends
from typing import List
from agents.video_search_agent import handle_video_search
from .shared.requests import VideoSearchRequest
from .shared.responses import VideoSearchResponse, VideoResult
from .shared.utils import convert_chat_history, setup_llm_and_embeddings
from .shared.exceptions import InvalidInputError, ServerError
from .config import get_chat_model, get_embeddings_model
from utils.logger import logger

router = APIRouter(
    prefix="/videos",
    tags=["videos"],
    responses={
        400: {"description": "Invalid request parameters"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    }
)


@router.post(
    "/",
    response_model=VideoSearchResponse,
    description="Search for videos based on query and chat history",
    responses={
        200: {
            "description": "Successfully retrieved videos",
            "model": VideoSearchResponse
        }
    }
)
async def video_search(
    request: VideoSearchRequest,
    chat_model=Depends(get_chat_model),
    embeddings_model=Depends(get_embeddings_model)
) -> VideoSearchResponse:
    """
    Search for videos based on query and chat history.
    
    This endpoint searches for relevant videos based on the provided query and
    chat history context. It returns a list of video results with metadata.
    
    Args:
        request: VideoSearchRequest containing search query and chat history
        chat_model: Language model for processing
        embeddings_model: Embeddings model for processing
        
    Returns:
        VideoSearchResponse containing list of video results
        
    Raises:
        InvalidInputError: If the query is empty
        ServerError: If an error occurs during video search
    """
    try:
        if not request.query:
            raise InvalidInputError("Query cannot be empty")

        history = convert_chat_history(request.chat_history)
        
        videos = await handle_video_search(
            query=request.query,
            history=history,
            llm=chat_model,
            embeddings=embeddings_model,
            optimization_mode=str(request.optimization_mode)
        )
        
        return VideoSearchResponse(
            videos=[VideoResult(**video) for video in videos]
        )
    except InvalidInputError:
        raise
    except Exception as e:
        logger.error("Error in video search: %s", str(e))
        raise ServerError() from e
