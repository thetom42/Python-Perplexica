"""
Base models for the application.

This module contains the core base models that are used throughout the application.
These models provide the foundation for more specific models while ensuring
consistent validation and behavior.
"""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class TimestampedModel(BaseModel):
    """
    Base model for entities with timestamps.
    
    Provides created_at and updated_at fields for tracking entity lifecycle.
    """
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True


class IdentifiedModel(TimestampedModel):
    """
    Base model for entities with ID and timestamps.
    
    Extends TimestampedModel to add an ID field for unique identification.
    """
    id: str = Field(..., description="Unique identifier")


class APIKeyModel(BaseModel):
    """
    Base model for API key configuration.
    
    Provides fields for API keys and endpoints with proper validation.
    """
    api_key: Optional[str] = Field(None, description="API key for authentication")
    api_endpoint: Optional[str] = Field(None, description="API endpoint URL")


class ChatBase(IdentifiedModel):
    """
    Base model for chat data.
    
    Extends IdentifiedModel to provide the base structure for chat entities.
    Includes ID and timestamp fields from parent classes.
    """
    pass


class MessageBase(IdentifiedModel):
    """
    Base model for chat messages.
    
    Extends IdentifiedModel to provide the base structure for message entities.
    Includes ID and timestamp fields from parent classes, plus message-specific fields.
    """
    chat_id: str = Field(..., description="ID of the associated chat")
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
