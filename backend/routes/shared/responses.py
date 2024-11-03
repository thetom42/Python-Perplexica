"""
Response models for the application.

This module contains all the response models used by the API endpoints.
These models ensure consistent response formats and provide clear
documentation of the API's output structure.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional
from datetime import datetime
from routes.shared.base import IdentifiedModel
from routes.shared.config import ModelInfo


class ImageResult(BaseModel):
    """Model for image search results."""
    url: HttpUrl = Field(..., description="URL of the image")
    title: str = Field(..., description="Title of the image")
    source: Optional[HttpUrl] = Field(None, description="Source of the image")


class VideoResult(BaseModel):
    """Model for video search results."""
    title: str = Field(..., description="Title of the video")
    url: HttpUrl = Field(..., description="URL of the video")
    thumbnail: Optional[HttpUrl] = Field(None, description="Thumbnail URL")
    description: Optional[str] = Field(None, description="Video description")
    duration: Optional[str] = Field(None, description="Video duration")
    channel: Optional[str] = Field(None, description="Channel name")


class StreamResponse(BaseModel):
    """Model for streaming response data."""
    type: str = Field(..., description="Type of the response")
    data: Any = Field(..., description="Response data")


class ModelsResponse(BaseModel):
    """Response model for available models."""
    chat_model_providers: Dict[str, List[ModelInfo]] = Field(
        ...,
        description="Available chat model providers and their models"
    )
    embedding_model_providers: Dict[str, List[ModelInfo]] = Field(
        ...,
        description="Available embedding model providers and their models"
    )


class ChatMessage(IdentifiedModel):
    """Model for chat messages."""
    chat_id: str = Field(..., description="Associated chat ID")
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")


class ChatDetailResponse(BaseModel):
    """Response model for detailed chat information."""
    chat: IdentifiedModel = Field(..., description="Chat information")
    messages: List[ChatMessage] = Field(..., description="Chat messages")


class ChatsListResponse(BaseModel):
    """Response model for list of chats."""
    chats: List[IdentifiedModel] = Field(..., description="List of chats")


class DeleteResponse(BaseModel):
    """Response model for deletion confirmation."""
    message: str = Field(..., description="Confirmation message")


class VideoSearchResponse(BaseModel):
    """Response model for video search results."""
    videos: List[VideoResult] = Field(..., description="List of video results")
