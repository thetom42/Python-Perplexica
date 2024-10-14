from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid
import json
from backend.db.models import Chat, Message
from backend.db.database import get_db
from sqlalchemy.orm import Session

router = APIRouter()

class ChatCreate(BaseModel):
    title: str
    focus_mode: bool = False

class ChatUpdate(BaseModel):
    title: Optional[str] = None
    focus_mode: Optional[bool] = None

class MessageCreate(BaseModel):
    content: str
    chat_id: str
    role: str

@router.post("/")
async def create_chat(chat: ChatCreate, db: Session = Depends(get_db)):
    chat_id = str(uuid.uuid4())
    new_chat = Chat(
        id=chat_id,
        title=chat.title,
        created_at=datetime.now(),
        focus_mode=chat.focus_mode
    )
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    return {"id": chat_id}

@router.get("/")
async def get_chats(db: Session = Depends(get_db)):
    chats_list = db.query(Chat).all()
    return chats_list

@router.get("/{chat_id}")
async def get_chat(chat_id: str, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

@router.put("/{chat_id}")
async def update_chat(chat_id: str, chat_update: ChatUpdate, db: Session = Depends(get_db)):
    update_data = chat_update.dict(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    for key, value in update_data.items():
        setattr(chat, key, value)
    db.commit()
    db.refresh(chat)
    return {"message": "Chat updated successfully"}

@router.delete("/{chat_id}")
async def delete_chat(chat_id: str, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    db.delete(chat)
    db.commit()
    return {"message": "Chat deleted successfully"}

@router.post("/{chat_id}/messages")
async def create_message(chat_id: str, message: MessageCreate, db: Session = Depends(get_db)):
    new_message = Message(
        id=str(uuid.uuid4()),
        content=message.content,
        chat_id=chat_id,
        role=message.role,
        created_at=datetime.now()
    )
    db.add(new_message)
    db.commit()
    db.refresh(new_message)
    return {"id": new_message.id}

@router.get("/{chat_id}/messages")
async def get_chat_messages(chat_id: str, db: Session = Depends(get_db)):
    messages_list = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at).all()
    return messages_list
