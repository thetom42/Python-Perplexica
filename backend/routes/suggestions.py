from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langchain.schema import HumanMessage, AIMessage
from backend.agents.suggestion_generator_agent import generate_suggestions
from backend.lib.providers import get_available_chat_model_providers
from backend.utils.logger import logger

router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: str

class SuggestionRequest(BaseModel):
    chat_history: List[ChatMessage]
    chat_model: Optional[str] = None
    chat_model_provider: Optional[str] = None

@router.post("/")
async def generate_suggestions_route(request: SuggestionRequest) -> Dict[str, List[str]]:
    try:
        chat_history = [
            HumanMessage(content=msg.content) if msg.role == 'user' else AIMessage(content=msg.content)
            for msg in request.chat_history
        ]

        chat_models = await get_available_chat_model_providers()
        provider = request.chat_model_provider or next(iter(chat_models))
        chat_model = request.chat_model or next(iter(chat_models[provider]))

        llm = None
        if provider in chat_models and chat_model in chat_models[provider]:
            llm = chat_models[provider][chat_model].model

        if not llm:
            raise HTTPException(status_code=500, detail="Invalid LLM model selected")

        suggestions = await generate_suggestions({"chat_history": chat_history}, llm)

        return {"suggestions": suggestions}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generating suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail="An error has occurred.")
