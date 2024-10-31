from typing import AsyncGenerator, Dict, Any
from langchain.schema import HumanMessage, AIMessage

from backend.lib.searxng import search_searxng

async def handle_image_search(
    query: str,
    history: list[HumanMessage | AIMessage],
    chat_model: Any,
    embeddings_model: Any,
    mode: str = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle image search requests.
    
    Args:
        query: The search query
        history: Chat history
        chat_model: The chat model to use
        embeddings_model: The embeddings model to use
        mode: Search mode (speed/balanced/quality)
        
    Yields:
        Dict containing response type and data
    """
    try:
        # Perform image search
        results = await search_searxng(query, "images")

        if not results or not results.get("results"):
            yield {
                "type": "error",
                "data": "No images found. Please try a different search query."
            }
            return

        # Format image results
        image_results = []
        for result in results["results"]:
            if "img_src" in result:
                image_results.append({
                    "url": result["img_src"],
                    "title": result.get("title", ""),
                    "source": result.get("source", "")
                })

        if not image_results:
            yield {
                "type": "error",
                "data": "No valid images found. Please try a different search query."
            }
            return

        yield {
            "type": "images",
            "data": image_results
        }

    except Exception as e:
        yield {
            "type": "error",
            "data": f"An error occurred during image search: {str(e)}"
        }
