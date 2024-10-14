import pytest
from sqlalchemy.orm import Session
from backend.db import crud, models
from backend.db.models import Base
from backend.db.database import engine

@pytest.fixture(scope="module")
def db():
    Base.metadata.create_all(bind=engine)
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()

def test_create_chat(db):
    chat = crud.create_chat(db, title="Test Chat", focus_mode=True)
    assert chat.id is not None
    assert chat.title == "Test Chat"
    assert chat.focus_mode is True

def test_get_chats(db):
    crud.create_chat(db, title="Chat 1", focus_mode=False)
    crud.create_chat(db, title="Chat 2", focus_mode=True)
    chats = crud.get_chats(db)
    assert len(chats) >= 2

def test_get_chat(db):
    chat = crud.create_chat(db, title="Test Chat", focus_mode=True)
    retrieved_chat = crud.get_chat(db, chat.id)
    assert retrieved_chat.id == chat.id
    assert retrieved_chat.title == chat.title

def test_update_chat(db):
    chat = crud.create_chat(db, title="Original Title", focus_mode=False)
    updated_chat = crud.update_chat(db, chat.id, title="Updated Title", focus_mode=True)
    assert updated_chat.title == "Updated Title"
    assert updated_chat.focus_mode is True

def test_delete_chat(db):
    chat = crud.create_chat(db, title="To Be Deleted", focus_mode=False)
    deleted_chat = crud.delete_chat(db, chat.id)
    assert deleted_chat.id == chat.id
    assert crud.get_chat(db, chat.id) is None

def test_create_message(db):
    chat = crud.create_chat(db, title="Test Chat", focus_mode=False)
    message = crud.create_message(db, chat_id=chat.id, content="Test message", role="user")
    assert message.id is not None
    assert message.content == "Test message"
    assert message.role == "user"

def test_get_chat_messages(db):
    chat = crud.create_chat(db, title="Test Chat", focus_mode=False)
    crud.create_message(db, chat_id=chat.id, content="Message 1", role="user")
    crud.create_message(db, chat_id=chat.id, content="Message 2", role="assistant")
    messages = crud.get_chat_messages(db, chat.id)
    assert len(messages) == 2
