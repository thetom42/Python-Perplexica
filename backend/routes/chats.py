from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from db.models import Chat, Message
from db.database import get_db
from utils.logger import logger

router = APIRouter()


@router.get("/", response_model=Dict[str, List[Any]])
async def get_chats(db: Session = Depends(get_db)):
    """Get all chats in reverse chronological order."""
    try:
        chats = db.query(Chat).all()
        chats.reverse()  # Reverse to match TypeScript behavior
        return {"chats": chats}
    except Exception as err:
        logger.error("Error in getting chats: %s", str(err))
        raise HTTPException(status_code=500, detail="An error has occurred.") from err


@router.get("/{chat_id}", response_model=Dict[str, Any])
async def get_chat(chat_id: str, db: Session = Depends(get_db)):
    """Get a specific chat and its messages."""
    try:
        chat = db.query(Chat).filter(Chat.id == chat_id).first()

        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat_messages = db.query(Message).filter(Message.chatId == chat_id).all()
        return {"chat": chat, "messages": chat_messages}
    except HTTPException:
        raise
    except Exception as err:
        logger.error("Error in getting chat: %s", str(err))
        raise HTTPException(status_code=500, detail="An error has occurred.") from err


@router.delete("/{chat_id}", response_model=Dict[str, str])
async def delete_chat(chat_id: str, db: Session = Depends(get_db)):
    """Delete a chat and all its messages."""
    try:
        chat = db.query(Chat).filter(Chat.id == chat_id).first()

        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        # Delete messages first due to foreign key constraint
        db.query(Message).filter(Message.chatId == chat_id).delete()
        db.query(Chat).filter(Chat.id == chat_id).delete()
        db.commit()

        return {"message": "Chat deleted successfully"}
    except HTTPException:
        raise
    except Exception as err:
        logger.error("Error in deleting chat: %s", str(err))
        raise HTTPException(status_code=500, detail="An error has occurred.") from err
