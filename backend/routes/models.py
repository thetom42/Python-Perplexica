from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal

class ChatMessage(BaseModel):
    role: str
    content: str

class ImageSearchRequest(BaseModel):
    query: str
    history: List[ChatMessage]
    mode: Literal["speed", "balanced", "quality"] = "balanced"

class ImageResult(BaseModel):
    url: str
    title: str
    source: Optional[str] = None

class StreamResponse(BaseModel):
    type: str
    data: Any

class SearchRequest(BaseModel):
    query: str
    history: List[ChatMessage]
    mode: Literal["speed", "balanced", "quality"] = "balanced"

class ModelInfo(BaseModel):
    name: str
    displayName: str

class ModelsResponse(BaseModel):
    chatModelProviders: Dict[str, List[ModelInfo]]
    embeddingModelProviders: Dict[str, List[ModelInfo]]
