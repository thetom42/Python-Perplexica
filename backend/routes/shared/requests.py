"""
Request models for the application.

This module contains all the request models used by the API endpoints.
These models ensure proper validation of incoming requests and provide
clear documentation of expected request formats.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from routes.shared.enums import OptimizationMode, Role, FocusMode
from routes.shared.config import ChatModel, EmbeddingModel


class ChatMessage(BaseModel):
    """
    Model for chat messages in requests.
    
    Attributes:
        role: Role of the message sender (user/assistant)
        content: Content of the message
    """
    role: Role = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")


class BaseLLMRequest(BaseModel):
    """
    Base model for LLM-based requests.
    
    Provides common fields needed for requests that use language models.
    
    Attributes:
        chat_model: Optional chat model configuration
        embedding_model: Optional embedding model configuration
        optimization_mode: Mode for optimizing the request processing
    """
    chat_model: Optional[ChatModel] = Field(
        None,
        description="Chat model configuration"
    )
    embedding_model: Optional[EmbeddingModel] = Field(
        None,
        description="Embedding model configuration"
    )
    optimization_mode: OptimizationMode = Field(
        default=OptimizationMode.BALANCED,
        description="Optimization mode for the request"
    )


class SearchRequest(BaseLLMRequest):
    """
    Model for search requests.
    
    Attributes:
        focus_mode: Type of search to perform
        query: Search query string
        history: Previous chat messages for context
    """
    focus_mode: FocusMode = Field(..., description="Type of search to perform")
    query: str = Field(..., description="Search query string")
    history: List[ChatMessage] = Field(
        default=[],
        description="Previous chat messages for context"
    )


class ImageSearchRequest(BaseLLMRequest):
    """
    Model for image search requests.
    
    Attributes:
        query: Image search query
        history: Previous chat messages for context
    """
    query: str = Field(..., description="Image search query")
    history: List[ChatMessage] = Field(
        default=[],
        description="Previous chat messages for context"
    )


class VideoSearchRequest(BaseLLMRequest):
    """
    Model for video search requests.
    
    Attributes:
        query: Video search query
        chat_history: Previous chat messages for context
    """
    query: str = Field(..., description="Video search query")
    chat_history: List[ChatMessage] = Field(
        default=[],
        description="Previous chat messages for context"
    )
