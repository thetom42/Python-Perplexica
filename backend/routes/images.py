from typing import List
from fastapi import APIRouter, Depends, HTTPException
from langchain.schema import HumanMessage, AIMessage
from agents.image_search_agent import handle_image_search
from routes.models import ImageSearchRequest, ImageResult, StreamResponse
from routes.config import get_chat_model, get_embeddings_model
from utils.logger import logger

router = APIRouter()

@router.post("/search", response_model=List[StreamResponse])
async def search_images(
    request: ImageSearchRequest,
    chat_model=Depends(get_chat_model),
    embeddings_model=Depends(get_embeddings_model)
):
    """Search for images based on the given query."""
    try:
        history = [
            HumanMessage(content=msg.content) if msg.role == "user"
            else AIMessage(content=msg.content)
            for msg in request.history
        ]
        responses = []
        async for response in handle_image_search(
            request.query,
            history,
            chat_model,
            embeddings_model,
            request.mode
        ):
            responses.append(response)
        return responses
    except Exception as e:
        logger.error("Error in image search: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
