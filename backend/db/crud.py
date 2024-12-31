from typing import List, Optional
import uuid
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import desc
from db.models import Chat, Message, RoleEnum

class ChatNotFoundError(Exception):
    """Raised when a chat is not found in the database"""
    pass

class DatabaseError(Exception):
    """Raised when a database operation fails"""
    pass

def create_chat(db: Session, title: str, focus_mode: bool) -> Chat:
    """Create a new chat with the given title and focus mode"""
    try:
        chat = Chat(title=title, focus_mode=focus_mode)
        db.add(chat)
        db.commit()
        db.refresh(chat)
        return chat
    except SQLAlchemyError as e:
        db.rollback()
        raise DatabaseError("Failed to create chat") from e

def get_chats(db: Session, limit: int = 100, offset: int = 0) -> List[Chat]:
    """Get a list of chats with pagination support"""
    try:
        return db.query(Chat)\
            .order_by(desc(Chat.created_at))\
            .offset(offset)\
            .limit(limit)\
            .all()
    except SQLAlchemyError as e:
        raise DatabaseError("Failed to retrieve chats") from e

def get_chat(db: Session, chat_id: str) -> Chat:
    """Get a single chat by its ID"""
    try:
        chat = db.query(Chat).filter(Chat.id == chat_id).first()
        if not chat:
            raise ChatNotFoundError(f"Chat with ID {chat_id} not found")
        return chat
    except SQLAlchemyError as e:
        raise DatabaseError("Failed to retrieve chat") from e

def update_chat(db: Session, chat_id: str, title: Optional[str] = None, 
               focus_mode: Optional[bool] = None) -> Chat:
    """Update chat attributes"""
    try:
        chat = get_chat(db, chat_id)
        if title is not None:
            chat.title = title
        if focus_mode is not None:
            chat.focus_mode = focus_mode
        db.commit()
        db.refresh(chat)
        return chat
    except SQLAlchemyError as e:
        db.rollback()
        raise DatabaseError("Failed to update chat") from e

def delete_chat(db: Session, chat_id: str) -> Chat:
    """Delete a chat by its ID"""
    try:
        chat = get_chat(db, chat_id)
        db.delete(chat)
        db.commit()
        return chat
    except SQLAlchemyError as e:
        db.rollback()
        raise DatabaseError("Failed to delete chat") from e

def create_message(db: Session, chat_id: str, content: str,
                  role: RoleEnum) -> Message:
    """Create a new message in a chat"""
    try:
        message = Message(
            chat_id=chat_id,
            content=content,
            role=role,
            message_id=str(uuid.uuid4())
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        return message
    except SQLAlchemyError as e:
        db.rollback()
        raise DatabaseError("Failed to create message") from e

def get_chat_messages(db: Session, chat_id: str, 
                     limit: int = 100, offset: int = 0) -> List[Message]:
    """Get messages for a specific chat with pagination"""
    try:
        return db.query(Message)\
            .filter(Message.chat_id == chat_id)\
            .order_by(desc(Message.created_at))\
            .offset(offset)\
            .limit(limit)\
            .all()
    except SQLAlchemyError as e:
        raise DatabaseError("Failed to retrieve messages") from e

def bulk_create_messages(db: Session, messages: List[dict]) -> List[Message]:
    """Create multiple messages in a single transaction"""
    try:
        # Add message_id to each message data
        for message in messages:
            message['message_id'] = str(uuid.uuid4())
        message_objects = [Message(**message) for message in messages]
        db.bulk_save_objects(message_objects)
        db.commit()
        return message_objects
    except SQLAlchemyError as e:
        db.rollback()
        raise DatabaseError("Failed to create messages") from e
