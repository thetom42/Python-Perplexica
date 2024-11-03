from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain.chat_models.openai import ChatOpenAI  # Fixed import
from providers import get_available_chat_model_providers, get_available_embedding_model_providers
from websocket.message_handler import search_handlers  # This will be fixed in message_handler.py
from utils.logger import logger
import json

router = APIRouter()

class ChatModel(BaseModel):
    provider: str
    model: str
    customOpenAIBaseURL: Optional[str] = None
    customOpenAIKey: Optional[str] = None

class EmbeddingModel(BaseModel):
    provider: str
    model: str

class ChatRequestBody(BaseModel):
    optimizationMode: str = 'balanced'
    focusMode: str
    chatModel: Optional[ChatModel] = None
    embeddingModel: Optional[EmbeddingModel] = None
    query: str
    history: List[Tuple[str, str]] = []

@router.post("/")
async def search(body: ChatRequestBody):
    try:
        if not body.focusMode or not body.query:
            raise HTTPException(status_code=400, detail="Missing focus mode or query")

        history: List[BaseMessage] = [
            HumanMessage(content=msg[1]) if msg[0] == 'human' else AIMessage(content=msg[1])
            for msg in body.history
        ]

        chat_model_providers = await get_available_chat_model_providers()
        embedding_model_providers = await get_available_embedding_model_providers()

        chat_model_provider = body.chatModel.provider if body.chatModel else next(iter(chat_model_providers))
        chat_model = body.chatModel.model if body.chatModel else next(iter(chat_model_providers[chat_model_provider]))

        embedding_model_provider = body.embeddingModel.provider if body.embeddingModel else next(iter(embedding_model_providers))
        embedding_model = body.embeddingModel.model if body.embeddingModel else next(iter(embedding_model_providers[embedding_model_provider]))

        llm: Optional[BaseChatModel] = None
        embeddings: Optional[Embeddings] = None

        if body.chatModel and body.chatModel.provider == 'custom_openai':
            if not body.chatModel.customOpenAIBaseURL or not body.chatModel.customOpenAIKey:
                raise HTTPException(status_code=400, detail="Missing custom OpenAI base URL or key")

            llm = ChatOpenAI(
                name=body.chatModel.model,
                api_key=body.chatModel.customOpenAIKey,
                temperature=0.7,
                base_url=body.chatModel.customOpenAIBaseURL,
            )
        elif chat_model_provider in chat_model_providers and chat_model in chat_model_providers[chat_model_provider]:
            llm = chat_model_providers[chat_model_provider][chat_model].model

        if embedding_model_provider in embedding_model_providers and embedding_model in embedding_model_providers[embedding_model_provider]:
            embeddings = embedding_model_providers[embedding_model_provider][embedding_model].model

        if not llm or not embeddings:
            raise HTTPException(status_code=400, detail="Invalid model selected")

        search_handler = search_handlers.get(body.focusMode)
        if not search_handler:
            raise HTTPException(status_code=400, detail="Invalid focus mode")

        async def event_stream():
            message = ""
            sources = []

            async for data in search_handler(body.query, history, llm, embeddings, body.optimizationMode):
                parsed_data = json.loads(data)
                if parsed_data["type"] == "response":
                    message += parsed_data["data"]
                    yield json.dumps({"type": "response", "data": parsed_data["data"]}) + "\n"
                elif parsed_data["type"] == "sources":
                    sources = parsed_data["data"]
                    yield json.dumps({"type": "sources", "data": sources}) + "\n"

            yield json.dumps({"type": "end", "data": {"message": message, "sources": sources}}) + "\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.error("Error in getting search results: %s", str(e))
        raise HTTPException(status_code=500, detail="An error has occurred.") from e
