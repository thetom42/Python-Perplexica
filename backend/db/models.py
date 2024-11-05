from datetime import datetime
from typing import Optional

from sqlalchemy import String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Chat(Base):
    __tablename__ = 'chats'

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    createdAt: Mapped[str] = mapped_column(String, nullable=False)
    focusMode: Mapped[str] = mapped_column(String, nullable=False)


class Message(Base):
    __tablename__ = 'messages'

    id: Mapped[str] = mapped_column(String, primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chatId: Mapped[str] = mapped_column(String, nullable=False)
    messageId: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String)  # enum: ['assistant', 'user']
    metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
