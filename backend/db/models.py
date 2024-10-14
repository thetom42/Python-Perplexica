"""
Database models module for the Perplexica backend.

This module defines the SQLAlchemy ORM models for the Chat and Message entities.
"""

from sqlalchemy import Column, Integer, String, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import JSON, DateTime
from datetime import datetime

Base = declarative_base()

class Chat(Base):
    """
    SQLAlchemy model for the chats table.

    Attributes:
        id (str): The primary key of the chat.
        title (str): The title of the chat.
        created_at (datetime): The creation timestamp of the chat.
        focus_mode (bool): The focus mode status of the chat.
        messages (relationship): Relationship to the Message model.
    """
    __tablename__ = "chats"

    id = Column(String, primary_key=True, index=True)  # type: str
    title = Column(String, index=True)  # type: str
    created_at = Column(DateTime, default=datetime.now)  # type: datetime
    focus_mode = Column(Boolean)  # type: bool

    messages = relationship("Message", back_populates="chat")

class Message(Base):
    """
    SQLAlchemy model for the messages table.

    Attributes:
        id (int): The primary key of the message.
        content (str): The content of the message.
        chat_id (str): The foreign key referencing the chat this message belongs to.
        message_id (str): A unique identifier for the message.
        role (str): The role of the message sender (e.g., user, assistant).
        metadata (JSON): Additional metadata for the message.
        chat (relationship): Relationship to the Chat model.
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)  # type: int
    content = Column(Text)  # type: str
    chat_id = Column(String, ForeignKey("chats.id"))  # type: str
    message_id = Column(String, index=True)  # type: str
    role = Column(String)  # type: str
    metadata = Column(JSON)  # type: str

    chat = relationship("Chat", back_populates="messages")
