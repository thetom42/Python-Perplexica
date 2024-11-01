"""
CRUD operations module for the Perplexica backend.

This module provides functions for Create, Read, Update, and Delete operations
on the Chat and Message models in the database.
"""

from sqlalchemy.orm import Session
from db.models import Chat, Message
from datetime import datetime
import uuid
import json
from typing import Optional, List

def create_chat(db: Session, title: str, focus_mode: bool) -> Chat:
    """
    Create a new chat in the database.

    Args:
        db (Session): The database session.
        title (str): The title of the chat.
        focus_mode (bool): Whether the chat is in focus mode.

    Returns:
        Chat: The created chat object.
    """
    db_chat = Chat(
        id=str(uuid.uuid4()),
        title=title,
        created_at=datetime.now(),
        focus_mode=focus_mode
    )
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return db_chat

def get_chats(db: Session, skip: int = 0, limit: int = 100) -> List[Chat]:
    """
    Retrieve a list of chats from the database.

    Args:
        db (Session): The database session.
        skip (int, optional): Number of records to skip. Defaults to 0.
        limit (int, optional): Maximum number of records to return. Defaults to 100.

    Returns:
        List[Chat]: A list of chat objects.
    """
    return db.query(Chat).offset(skip).limit(limit).all()

def get_chat(db: Session, chat_id: str) -> Optional[Chat]:
    """
    Retrieve a specific chat from the database by its ID.

    Args:
        db (Session): The database session.
        chat_id (str): The ID of the chat to retrieve.

    Returns:
        Optional[Chat]: The chat object if found, None otherwise.
    """
    return db.query(Chat).filter(Chat.id == chat_id).first()

def update_chat(db: Session, chat_id: str, title: Optional[str] = None, focus_mode: Optional[bool] = None) -> Optional[Chat]:
    """
    Update a chat in the database.

    Args:
        db (Session): The database session.
        chat_id (str): The ID of the chat to update.
        title (Optional[str], optional): The new title for the chat. Defaults to None.
        focus_mode (Optional[bool], optional): The new focus mode status. Defaults to None.

    Returns:
        Optional[Chat]: The updated chat object if found, None otherwise.
    """
    db_chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if db_chat:
        if title is not None:
            db_chat.title = title
        if focus_mode is not None:
            db_chat.focus_mode = focus_mode
        db.commit()
        db.refresh(db_chat)
    return db_chat

def delete_chat(db: Session, chat_id: str) -> Optional[Chat]:
    """
    Delete a chat from the database.

    Args:
        db (Session): The database session.
        chat_id (str): The ID of the chat to delete.

    Returns:
        Optional[Chat]: The deleted chat object if found, None otherwise.
    """
    db_chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if db_chat:
        db.delete(db_chat)
        db.commit()
    return db_chat

def create_message(db: Session, chat_id: str, content: str, role: str, metadata: Optional[dict] = None) -> Message:
    """
    Create a new message in the database.

    Args:
        db (Session): The database session.
        chat_id (str): The ID of the chat the message belongs to.
        content (str): The content of the message.
        role (str): The role of the message sender.
        metadata (Optional[dict], optional): Additional metadata for the message. Defaults to None.

    Returns:
        Message: The created message object.
    """
    try:
        db_message = Message(
            chat_id=chat_id,
            content=content,
            role=role,
            message_id=str(uuid.uuid4()),
            metadata=json.dumps(metadata) if metadata else None
        )
        db.add(db_message)
        db.commit()
        db.refresh(db_message)
        return db_message
    except Exception as e:
        print(f"Error creating message: {e}")
        return None

def get_chat_messages(db: Session, chat_id: str) -> List[Message]:
    """
    Retrieve all messages for a specific chat from the database.

    Args:
        db (Session): The database session.
        chat_id (str): The ID of the chat to retrieve messages for.

    Returns:
        List[Message]: A list of message objects for the specified chat.
    """
    return db.query(Message).filter(Message.chat_id == chat_id).all()
