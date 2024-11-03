"""
Chat route handlers.

This module provides endpoints for managing chat sessions and messages,
including retrieving, creating, and deleting chats.
"""

from fastapi import APIRouter, HTTPException, Depends, Path
from typing import List
from sqlalchemy.orm import Session
from db.database import get_db
from db.models import Chat, Message
from utils.logger import logger
from routes.shared.exceptions import ServerError
from routes.shared.base import ChatBase, MessageBase
from routes.shared.responses import (
    ChatDetailResponse,
    ChatsListResponse,
    DeleteResponse
)

router = APIRouter(
    prefix="/chats",
    tags=["chats"],
    responses={
        404: {"description": "Chat not found"},
        500: {"description": "Internal server error"},
    }
)


def convert_chat(chat: Chat) -> ChatBase:
    """Convert database Chat model to response model."""
    return ChatBase(
        id=str(chat.id),
        created_at=chat.created_at,
        updated_at=chat.updated_at
    )


def convert_message(message: Message) -> MessageBase:
    """Convert database Message model to response model."""
    return MessageBase(
        id=str(message.id),
        chat_id=str(message.chat_id),
        role=message.role,
        content=message.content,
        created_at=message.created_at
    )


@router.get(
    "/",
    response_model=ChatsListResponse,
    description="Get all chats",
    responses={
        200: {
            "description": "List of all chats",
            "model": ChatsListResponse
        }
    }
)
async def get_chats(
    db: Session = Depends(get_db)
) -> ChatsListResponse:
    """
    Retrieve all chats ordered by creation date.
    
    Args:
        db: Database session
        
    Returns:
        ChatsListResponse containing list of chats
        
    Raises:
        ServerError: If database operation fails
    """
    try:
        chats = db.query(Chat).order_by(Chat.created_at.desc()).all()
        return ChatsListResponse(
            chats=[convert_chat(chat) for chat in chats]
        )
    except Exception as e:
        logger.error("Error in getting chats: %s", str(e))
        raise ServerError() from e


@router.get(
    "/{chat_id}",
    response_model=ChatDetailResponse,
    description="Get chat by ID",
    responses={
        200: {
            "description": "Chat details with messages",
            "model": ChatDetailResponse
        }
    }
)
async def get_chat(
    chat_id: str = Path(..., description="ID of the chat to retrieve"),
    db: Session = Depends(get_db)
) -> ChatDetailResponse:
    """
    Retrieve a specific chat and its messages.
    
    Args:
        chat_id: ID of the chat to retrieve
        db: Database session
        
    Returns:
        ChatDetailResponse containing chat details and messages
        
    Raises:
        HTTPException: If chat is not found
        ServerError: If database operation fails
    """
    try:
        chat = db.query(Chat).filter(Chat.id == chat_id).first()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat_messages = (
            db.query(Message)
            .filter(Message.chat_id == chat_id)
            .order_by(Message.created_at)
            .all()
        )
        return ChatDetailResponse(
            chat=convert_chat(chat),
            messages=[convert_message(msg) for msg in chat_messages]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in getting chat: %s", str(e))
        raise ServerError() from e


@router.delete(
    "/{chat_id}",
    response_model=DeleteResponse,
    description="Delete a chat",
    responses={
        200: {
            "description": "Chat deleted successfully",
            "model": DeleteResponse
        }
    }
)
async def delete_chat(
    chat_id: str = Path(..., description="ID of the chat to delete"),
    db: Session = Depends(get_db)
) -> DeleteResponse:
    """
    Delete a chat and all its messages.
    
    Args:
        chat_id: ID of the chat to delete
        db: Database session
        
    Returns:
        DeleteResponse confirming deletion
        
    Raises:
        HTTPException: If chat is not found
        ServerError: If database operation fails
    """
    try:
        chat = db.query(Chat).filter(Chat.id == chat_id).first()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        db.query(Message).filter(Message.chat_id == chat_id).delete()
        db.delete(chat)
        db.commit()
        return DeleteResponse(message="Chat deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in deleting chat: %s", str(e))
        raise ServerError() from e
