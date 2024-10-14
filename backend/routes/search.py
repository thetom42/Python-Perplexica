from fastapi import APIRouter, HTTPException, Depends
from langchain.schema import BaseMessage
from typing import List
from pydantic import BaseModel
from backend.agents.web_search_agent import handle_web_search
from backend.lib.providers import get_chat_model, get_embeddings_model
from backend.utils.logger import logger

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    history: List[dict]

class SearchResponse(BaseModel):
    type: str
    data: str

@router.post("/web", response_model=SearchResponse)
async def web_search(request: SearchRequest):
    try:
        llm = get_chat_model()
        embeddings = get_embeddings_model()

        # Convert dict history to BaseMessage objects
        history = [BaseMessage(**msg) for msg in request.history]

        result = await handle_web_search(request.query, history, llm, embeddings)
        return SearchResponse(**result)
    except Exception as e:
        logger.error("Error in web search: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during the web search") from e

# Add more search endpoints (academic, image, etc.) as needed
