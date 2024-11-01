from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langchain.schema import HumanMessage, AIMessage
from agents.video_search_agent import handle_video_search
from lib.providers import get_available_chat_model_providers
from utils.logger import logger
from langchain.embeddings import OpenAIEmbeddings  # Add this import

router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: str

class VideoSearchRequest(BaseModel):
    query: str
    chat_history: List[ChatMessage]
    chat_model_provider: Optional[str] = None
    chat_model: Optional[str] = None

@router.post("/")
async def video_search(request: VideoSearchRequest) -> Dict[str, List[Dict[str, Any]]]:
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

        embeddings = OpenAIEmbeddings()
        videos = await handle_video_search(
            query=request.query,
            history=chat_history,
            llm=llm,
            embeddings=embeddings,
            optimization_mode="balanced"
        )

        return {"videos": videos}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in video search: %s", str(e))
        raise HTTPException(status_code=500, detail="An error has occurred.") from e
