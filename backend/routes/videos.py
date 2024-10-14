from fastapi import APIRouter, Depends
from langchain.schema import BaseMessage
from typing import List
from backend.agents.video_search_agent import handle_video_search
from backend.lib.providers import get_chat_model, get_embeddings_model

router = APIRouter()

@router.post("/search")
async def video_search(query: str, history: List[BaseMessage]):
    llm = get_chat_model()
    embeddings = get_embeddings_model()
    result = await handle_video_search(query, history, llm, embeddings)
    return result
