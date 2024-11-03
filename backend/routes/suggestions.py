"""
Suggestions route handlers.

This module provides endpoints for generating suggestions based on chat history
and user interactions.
"""

from fastapi import APIRouter
from typing import Dict, List
from .shared.requests import BaseLLMRequest
from .shared.responses import SuggestionResponse
from .shared.utils import convert_chat_history, setup_llm_and_embeddings
from .shared.exceptions import ServerError
from agents.suggestion_generator_agent import handle_suggestion_generation
from utils.logger import logger

router = APIRouter(
    prefix="/suggestions",
    tags=["suggestions"],
    responses={
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    }
)


@router.post(
    "/",
    response_model=SuggestionResponse,
    description="Generate suggestions based on chat history",
    responses={
        200: {
            "description": "Successfully generated suggestions",
            "model": SuggestionResponse
        }
    }
)
async def generate_suggestions(request: BaseLLMRequest) -> SuggestionResponse:
    """
    Generate suggestions based on chat history.
    
    This endpoint analyzes the provided chat history and generates relevant
    suggestions for continuing the conversation or exploring related topics.
    
    Args:
        request: BaseLLMRequest containing chat history and model configuration
        
    Returns:
        SuggestionResponse containing list of suggestions
        
    Raises:
        ServerError: If an error occurs during suggestion generation
    """
    try:
        chat_history = convert_chat_history(request.history)
        llm, embeddings = await setup_llm_and_embeddings(
            request.chat_model,
            request.embedding_model
        )

        suggestions = await handle_suggestion_generation(
            history=chat_history,
            llm=llm,
            embeddings=embeddings,
            optimization_mode=str(request.optimization_mode)  # Convert enum to string
        )

        return SuggestionResponse(suggestions=suggestions)

    except Exception as e:
        logger.error("Error in generating suggestions: %s", str(e))
        raise ServerError() from e
