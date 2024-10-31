from datetime import datetime
from typing import List, Optional

from sqlalchemy import ForeignKey, Integer, String, Text, JSON, DateTime, Boolean
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Chat(Base):
    """
    SQLAlchemy model for the chats table.

    Attributes:
        id (int): The primary key of the chat.
        title (str): The title of the chat.
        created_at (datetime): The timestamp when the chat was created.
        updated_at (datetime): The timestamp when the chat was last updated.
        messages (List[Message]): The messages in this chat.
    """
    __tablename__ = 'chats'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages: Mapped[List["Message"]] = relationship("Message", back_populates="chat", cascade="all, delete-orphan")


class Message(Base):
    """
    SQLAlchemy model for the messages table.

    Attributes:
        id (int): The primary key of the message.
        chat_id (int): The foreign key referencing the chat this message belongs to.
        role (str): The role of the message sender (e.g., 'user', 'assistant').
        content (str): The content of the message.
        created_at (datetime): The timestamp when the message was created.
        updated_at (datetime): The timestamp when the message was last updated.
        sources (JSON): JSON field containing source information.
        chat (Chat): The chat this message belongs to.
    """
    __tablename__ = 'messages'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(Integer, ForeignKey('chats.id'), nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    sources: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    chat: Mapped["Chat"] = relationship("Chat", back_populates="messages")


class Config(Base):
    """
    SQLAlchemy model for the config table.

    Attributes:
        id (int): The primary key of the config.
        key (str): The configuration key.
        value (str): The configuration value.
        is_secret (bool): Whether this config value is a secret.
    """
    __tablename__ = 'config'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    is_secret: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
