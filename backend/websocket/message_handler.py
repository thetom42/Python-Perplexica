import json
from fastapi import WebSocket
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from agents import (
    handle_web_search,
    handle_academic_search,
    handle_image_search,
    handle_reddit_search,
    handle_video_search,
    handle_wolfram_alpha_search,
    handle_writing_assistance,
    handle_youtube_search,
)
from providers import get_chat_model, get_embeddings_model

#__all__ = ['search_handlers', 'handle_message', 'convert_to_base_message']

# Map focus modes to their respective handler functions
search_handlers = {
    "web": handle_web_search,
    "academic": handle_academic_search,
    "image": handle_image_search,
    "reddit": handle_reddit_search,
    "video": handle_video_search,
    "wolfram_alpha": handle_wolfram_alpha_search,
    "writing": handle_writing_assistance,
    "youtube": handle_youtube_search,
}

async def handle_message(websocket: WebSocket, message: str):
    try:
        data = json.loads(message)
        message_type = data.get("type")
        query = data.get("query")
        history = [convert_to_base_message(msg) for msg in data.get("history", [])]

        llm = get_chat_model()
        embeddings = get_embeddings_model()

        if message_type == "web_search":
            result = await handle_web_search(query, history, llm, embeddings)
        elif message_type == "academic_search":
            result = await handle_academic_search(query, history, llm, embeddings)
        elif message_type == "image_search":
            result = await handle_image_search(query, history, llm, embeddings)
        elif message_type == "reddit_search":
            result = await handle_reddit_search(query, history, llm, embeddings)
        elif message_type == "video_search":
            result = await handle_video_search(query, history, llm, embeddings)
        elif message_type == "wolfram_alpha":
            result = await handle_wolfram_alpha_search(query, history, llm, embeddings)
        elif message_type == "writing_assistance":
            result = await handle_writing_assistance(query, history, llm, embeddings)
        elif message_type == "youtube_search":
            result = await handle_youtube_search(query, history, llm, embeddings)
        else:
            result = {"error": "Unknown message type"}

        await websocket.send_text(json.dumps(result))
    except Exception as e:
        error_message = {"error": f"An error occurred: {str(e)}"}
        await websocket.send_text(json.dumps(error_message))

def convert_to_base_message(message: dict) -> BaseMessage:
    if message["role"] == "human":
        return HumanMessage(content=message["content"])
    elif message["role"] == "ai":
        return AIMessage(content=message["content"])
    else:
        raise ValueError(f"Unknown message role: {message['role']}")
