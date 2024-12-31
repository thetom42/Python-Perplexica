from datetime import datetime
from typing import Optional, List
from enum import Enum
import uuid

from sqlalchemy import String, Text, JSON, DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RoleEnum(str, Enum):
    ASSISTANT = 'assistant'
    USER = 'user'

class Chat(Base):
    """Represents a chat conversation"""
    __tablename__ = 'chats'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    focus_mode: Mapped[bool] = mapped_column(default=False, nullable=False)
    
    # Relationship to messages
    messages: Mapped[List['Message']] = relationship(back_populates='chat', cascade='all, delete-orphan')

    __table_args__ = (
        Index('idx_chat_created_at', 'created_at'),
    )

class Message(Base):
    """Represents a message within a chat"""
    __tablename__ = 'messages'

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chat_id: Mapped[str] = mapped_column(String(36), ForeignKey('chats.id', ondelete='CASCADE'), nullable=False)
    message_id: Mapped[str] = mapped_column(String(36), nullable=False)
    role: Mapped[RoleEnum] = mapped_column(String(20), nullable=False)
    message_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship to chat
    chat: Mapped['Chat'] = relationship(back_populates='messages')

    __table_args__ = (
        Index('idx_message_chat_id', 'chat_id'),
        Index('idx_message_created_at', 'created_at'),
    )
