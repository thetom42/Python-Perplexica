"""
Image search route handlers.

This module provides endpoints for searching images based on queries and chat history.
"""

from fastapi import APIRouter, Depends
from typing import List
from langchain.schema import HumanMessage, AIMessage
from agents.image_search_agent import handle_image_search
from .shared.requests import ImageSearchRequest
from .shared.responses import StreamResponse
from .shared.utils import convert_chat_history, setup_llm_and_embeddings
from .shared.exceptions import InvalidInputError, ServerError
from .config import get_chat_model, get_embeddings_model
from utils.logger import logger

router = APIRouter(
    prefix="/images",
    tags=["images"],
    responses={
        400: {"description": "Invalid request parameters"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    }
)


@router.post(
    "/search",
    response_model=List[StreamResponse],
    description="Search for images based on query",
    responses={
        200: {
            "description": "List of image search results",
            "model": List[StreamResponse]
        }
    }
)
async def search_images(
    request: ImageSearchRequest,
    chat_model=Depends(get_chat_model),
    embeddings_model=Depends(get_embeddings_model)
) -> List[StreamResponse]:
    """
    Search for images based on the given query.
    
    This endpoint performs an image search using the provided query and chat history
    context. It returns a list of image results with metadata.
    
    Args:
        request: ImageSearchRequest containing search parameters
        chat_model: Language model for processing
        embeddings_model: Embeddings model for processing
        
    Returns:
        List of StreamResponse containing image results
        
    Raises:
        InvalidInputError: If the query is empty
        ServerError: If an error occurs during image search
    """
    try:
        if not request.query:
            raise InvalidInputError("Query cannot be empty")

        history = convert_chat_history(request.history)
        
        responses = []
        async for response in handle_image_search(
            request.query,
            history,
            chat_model,
            embeddings_model,
            str(request.optimization_mode)
        ):
            responses.append(StreamResponse(**response))
            
        return responses
    except InvalidInputError:
        raise
    except Exception as e:
        logger.error("Error in image search: %s", str(e))
        raise ServerError() from e
