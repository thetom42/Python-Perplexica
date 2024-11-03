"""
Search route handlers.

This module provides the search endpoints that support different types of searches
(web, academic, image, etc.) with streaming responses.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from .shared.enums import FocusMode, OptimizationMode
from .shared.requests import SearchRequest
from .shared.responses import StreamResponse
from .shared.utils import convert_chat_history, setup_llm_and_embeddings
from .shared.exceptions import InvalidInputError, ServerError
from websocket.message_handler import search_handlers
from utils.logger import logger
import json

router = APIRouter(
    prefix="/search",
    tags=["search"],
    responses={
        400: {"description": "Invalid request parameters"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    }
)


@router.post(
    "/",
    response_class=StreamingResponse,
    description="Perform a search based on the specified focus mode and query",
    responses={
        200: {
            "description": "Successful search",
            "content": {
                "text/event-stream": {
                    "example": 'data: {"type": "response", "data": "Search result"}\n'
                }
            }
        }
    }
)
async def search(body: SearchRequest) -> StreamingResponse:
    """
    Perform a search based on the specified focus mode and query.
    
    The search is performed asynchronously and results are streamed back to the client
    using server-sent events (SSE). The response includes both the search results and
    any relevant sources.
    
    Args:
        body: SearchRequest containing the search parameters
        
    Returns:
        StreamingResponse: Server-sent events stream containing search results
        
    Raises:
        InvalidInputError: If the query is empty or focus mode is invalid
        ServerError: If an internal error occurs during search
    """
    try:
        if not body.query:
            raise InvalidInputError("Query cannot be empty")

        history = convert_chat_history(body.history)
        llm, embeddings = await setup_llm_and_embeddings(
            body.chat_model,
            body.embedding_model
        )

        search_handler = search_handlers.get(str(body.focus_mode))
        if not search_handler:
            raise InvalidInputError(f"Invalid focus mode: {body.focus_mode}")

        async def event_stream():
            message = ""
            sources = []

            async for data in search_handler(
                body.query,
                history,
                llm,
                embeddings,
                str(body.optimization_mode)  # Convert enum to string
            ):
                parsed_data = json.loads(data)
                response = StreamResponse(
                    type=parsed_data["type"],
                    data=parsed_data["data"]
                )
                
                if response.type == "response":
                    message += response.data
                    yield json.dumps(response.dict()) + "\n"
                elif response.type == "sources":
                    sources = response.data
                    yield json.dumps(response.dict()) + "\n"

            yield json.dumps(
                StreamResponse(
                    type="end",
                    data={
                        "message": message,
                        "sources": sources
                    }
                ).dict()
            ) + "\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream"
        )

    except InvalidInputError:
        raise
    except Exception as e:
        logger.error("Error in search: %s", str(e))
        raise ServerError() from e
