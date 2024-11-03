"""
Configuration models for the application.

This module contains models related to configuration, including model settings,
API keys, and other configurable parameters.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional
from routes.shared.base import APIKeyModel


class ChatModel(BaseModel):
    """
    Configuration for chat models.
    
    Attributes:
        provider: The provider of the chat model (e.g., OpenAI, Anthropic)
        model: The specific model to use
        custom_openai_base_url: Optional custom OpenAI API endpoint
        custom_openai_key: Optional custom OpenAI API key
    """
    provider: str = Field(..., description="Provider of the chat model")
    model: str = Field(..., description="Name of the chat model")
    custom_openai_base_url: Optional[str] = Field(
        None,
        description="Custom OpenAI base URL"
    )
    custom_openai_key: Optional[str] = Field(
        None,
        description="Custom OpenAI API key"
    )


class EmbeddingModel(BaseModel):
    """
    Configuration for embedding models.
    
    Attributes:
        provider: The provider of the embedding model
        model: The specific model to use
    """
    provider: str = Field(..., description="Provider of the embedding model")
    model: str = Field(..., description="Name of the embedding model")


class ModelInfo(BaseModel):
    """
    Information about an available model.
    
    Attributes:
        name: Internal name of the model
        display_name: Human-readable name for display
    """
    name: str = Field(..., description="Internal name of the model")
    display_name: str = Field(..., description="Display name of the model")


class APIConfig(APIKeyModel):
    """
    API configuration settings.
    
    Attributes:
        openai_api_key: OpenAI API key
        ollama_api_url: Ollama API endpoint URL
        anthropic_api_key: Anthropic API key
        groq_api_key: Groq API key
    """
    openai_api_key: str = Field(..., description="OpenAI API key")
    ollama_api_url: str = Field(..., description="Ollama API endpoint URL")
    anthropic_api_key: str = Field(..., description="Anthropic API key")
    groq_api_key: str = Field(..., description="Groq API key")
