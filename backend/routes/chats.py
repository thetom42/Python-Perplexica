from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from backend.db.database import get_db
from backend.db.models import Chat, Message
from backend.utils.logger import logger

router = APIRouter()

@router.get("/")
async def get_chats(db: Session = Depends(get_db)) -> Dict[str, List[Dict[str, Any]]]:
    try:
        chats = db.query(Chat).order_by(Chat.created_at.desc()).all()
        return {"chats": [chat.to_dict() for chat in chats]}
    except Exception as e:
        logger.error("Error in getting chats: %s", str(e))
        raise HTTPException(status_code=500, detail="An error has occurred.") from e

@router.get("/{chat_id}")
async def get_chat(chat_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    try:
        chat = db.query(Chat).filter(Chat.id == chat_id).first()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat_messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at).all()
        return {"chat": chat.to_dict(), "messages": [message.to_dict() for message in chat_messages]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in getting chat: %s", str(e))
        raise HTTPException(status_code=500, detail="An error has occurred.") from e

@router.delete("/{chat_id}")
async def delete_chat(chat_id: str, db: Session = Depends(get_db)) -> Dict[str, str]:
    try:
        chat = db.query(Chat).filter(Chat.id == chat_id).first()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        db.query(Message).filter(Message.chat_id == chat_id).delete()
        db.delete(chat)
        db.commit()
        return {"message": "Chat deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in deleting chat: %s", str(e))
        raise HTTPException(status_code=500, detail="An error has occurred.") from e

# Note: The create_chat, update_chat, create_message, and get_chat_messages endpoints
# were not present in the original TypeScript implementation, so they have been removed.
