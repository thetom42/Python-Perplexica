"""
Tests for the chats route.

This module contains tests for the chat management endpoints, including
retrieving and deleting chats.
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from sqlalchemy.orm import Session

from main import app
from routes.chats import router as chats_router
from db.models import Chat, Message

app.include_router(chats_router)

client = TestClient(app)


def create_mock_chat() -> dict:
    """Create a mock chat for testing."""
    return {
        "id": "test-chat-id",
        "title": "Test Chat",
        "createdAt": datetime.utcnow().isoformat(),
        "focusMode": "default"
    }


def create_mock_message(chat_id: str = "test-chat-id") -> dict:
    """Create a mock message for testing."""
    return {
        "id": "test-message-id",
        "chatId": chat_id,
        "messageId": "test-message-id",
        "role": "user",
        "content": "Test message",
        "metadata": None
    }


@pytest.mark.asyncio
async def test_get_chats():
    """Test getting all chats."""
    mock_chat = create_mock_chat()

    with patch('routes.chats.get_db') as mock_db:
        mock_session = MagicMock(spec=Session)
        mock_chat_obj = MagicMock(spec=Chat)
        for key, value in mock_chat.items():
            setattr(mock_chat_obj, key, value)

        mock_session.query.return_value.all.return_value = [mock_chat_obj]
        mock_db.return_value.__next__.return_value = mock_session

        response = client.get("/chats")

        assert response.status_code == 200
        result = response.json()
        assert "chats" in result
        assert len(result["chats"]) == 1
        assert result["chats"][0]["id"] == mock_chat["id"]
        assert result["chats"][0]["title"] == mock_chat["title"]
        assert result["chats"][0]["focusMode"] == mock_chat["focusMode"]


@pytest.mark.asyncio
async def test_get_chat_by_id():
    """Test getting a specific chat by ID."""
    mock_chat = create_mock_chat()
    mock_message = create_mock_message()

    with patch('routes.chats.get_db') as mock_db:
        mock_session = MagicMock(spec=Session)

        # Mock chat object
        mock_chat_obj = MagicMock(spec=Chat)
        for key, value in mock_chat.items():
            setattr(mock_chat_obj, key, value)

        # Mock message object
        mock_message_obj = MagicMock(spec=Message)
        for key, value in mock_message.items():
            setattr(mock_message_obj, key, value)

        mock_session.query.return_value.filter.return_value.first.return_value = mock_chat_obj
        mock_session.query.return_value.filter.return_value.all.return_value = [mock_message_obj]
        mock_db.return_value.__next__.return_value = mock_session

        response = client.get(f"/chats/{mock_chat['id']}")

        assert response.status_code == 200
        result = response.json()
        assert "chat" in result
        assert "messages" in result
        assert result["chat"]["id"] == mock_chat["id"]
        assert len(result["messages"]) == 1
        assert result["messages"][0]["id"] == mock_message["id"]
        assert result["messages"][0]["chatId"] == mock_message["chatId"]


@pytest.mark.asyncio
async def test_get_chat_not_found():
    """Test getting a non-existent chat."""
    with patch('routes.chats.get_db') as mock_db:
        mock_session = MagicMock(spec=Session)
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_db.return_value.__next__.return_value = mock_session

        response = client.get("/chats/non-existent-id")

        assert response.status_code == 404
        assert response.json()["detail"] == "Chat not found"


@pytest.mark.asyncio
async def test_delete_chat():
    """Test deleting a chat."""
    mock_chat = create_mock_chat()

    with patch('routes.chats.get_db') as mock_db:
        mock_session = MagicMock(spec=Session)
        mock_chat_obj = MagicMock(spec=Chat)
        for key, value in mock_chat.items():
            setattr(mock_chat_obj, key, value)

        mock_session.query.return_value.filter.return_value.first.return_value = mock_chat_obj
        mock_db.return_value.__next__.return_value = mock_session

        response = client.delete(f"/chats/{mock_chat['id']}")

        assert response.status_code == 200
        result = response.json()
        assert result["message"] == "Chat deleted successfully"

        # Verify delete operations were called
        mock_session.query.return_value.filter.return_value.delete.assert_called()
        mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_chat_not_found():
    """Test deleting a non-existent chat."""
    with patch('routes.chats.get_db') as mock_db:
        mock_session = MagicMock(spec=Session)
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_db.return_value.__next__.return_value = mock_session

        response = client.delete("/chats/non-existent-id")

        assert response.status_code == 404
        assert response.json()["detail"] == "Chat not found"


@pytest.mark.asyncio
async def test_database_error():
    """Test handling of database errors."""
    with patch('routes.chats.get_db') as mock_db:
        mock_session = MagicMock(spec=Session)
        mock_session.query.side_effect = Exception("Database error")
        mock_db.return_value.__next__.return_value = mock_session

        response = client.get("/chats")

        assert response.status_code == 500
        assert response.json()["detail"] == "An error has occurred."
