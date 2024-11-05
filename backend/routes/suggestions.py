"""
Suggestions route handlers.

This module provides endpoints for generating suggestions based on chat history.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain.schema import HumanMessage, AIMessage
from agents.suggestion_generator_agent import handle_suggestion_generation
from providers import get_available_chat_model_providers
from utils.logger import logger


class SuggestionsRequest(BaseModel):
    chat_history: List[Dict[str, str]]
    chat_model: Optional[str] = None
    chat_model_provider: Optional[str] = None


class SuggestionsResponse(BaseModel):
    suggestions: List[str]


router = APIRouter(
    tags=["suggestions"],
    responses={
        500: {"description": "Internal server error"},
    }
)


@router.post(
    "/",
    response_model=SuggestionsResponse,
    description="Generate suggestions based on chat history",
    responses={
        200: {
            "description": "Successfully generated suggestions",
            "model": SuggestionsResponse
        }
    }
)
async def generate_suggestions(request: SuggestionsRequest) -> SuggestionsResponse:
    """
    Generate suggestions based on chat history.
    
    Args:
        request: SuggestionsRequest containing chat history and model configuration
        
    Returns:
        SuggestionsResponse containing list of suggestions
        
    Raises:
        HTTPException: If an error occurs during suggestion generation
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

        # Generate suggestions
        suggestions = await handle_suggestion_generation(
            {"chat_history": chat_history},
            llm
        )

        return SuggestionsResponse(suggestions=suggestions)
    except Exception as e:
        logger.error("Error in generating suggestions: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail="An error has occurred."
        )
