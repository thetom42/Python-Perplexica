from fastapi import APIRouter, Depends
from langchain.schema import BaseMessage
from typing import List
from backend.agents.suggestion_generator_agent import handle_suggestion_generation
from backend.lib.providers import get_chat_model, get_embeddings_model

router = APIRouter()

@router.post("/generate")
async def generate_suggestions(query: str, history: List[BaseMessage]):
    llm = get_chat_model()
    embeddings = get_embeddings_model()
    result = await handle_suggestion_generation(query, history, llm, embeddings)
    return result
