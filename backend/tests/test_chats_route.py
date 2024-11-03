"""
Tests for the chats route.

This module contains tests for the chat management endpoints, including
retrieving, creating, and deleting chats.
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app
from routes.chats import router as chats_router
from routes.shared.responses import (
    ChatDetailResponse,
    ChatsListResponse,
    DeleteResponse,
    ChatBase,
    MessageBase
)

app.include_router(chats_router)

client = TestClient(app)


def create_mock_chat() -> dict:
    """Create a mock chat for testing."""
    now = datetime.utcnow()
    return {
        "id": "test-chat-id",
        "created_at": now,
        "updated_at": now
    }


def create_mock_message(chat_id: str = "test-chat-id") -> dict:
    """Create a mock message for testing."""
    return {
        "id": "test-message-id",
        "chat_id": chat_id,
        "role": "user",
        "content": "Test message",
        "created_at": datetime.utcnow()
    }


@pytest.mark.asyncio
async def test_get_chats():
    """Test getting all chats."""
    mock_chat = create_mock_chat()

    with patch('routes.chats.get_db') as mock_db:
        mock_session = MagicMock()
        mock_session.query().order_by().all.return_value = [
            type('Chat', (), mock_chat)
        ]
        mock_db.return_value = mock_session

        response = client.get("/chats")
        
        assert response.status_code == 200
        result = response.json()
        assert "chats" in result
        assert len(result["chats"]) == 1
        assert result["chats"][0]["id"] == mock_chat["id"]


@pytest.mark.asyncio
async def test_get_chat_by_id():
    """Test getting a specific chat by ID."""
    mock_chat = create_mock_chat()
    mock_message = create_mock_message()

    with patch('routes.chats.get_db') as mock_db:
        mock_session = MagicMock()
        mock_session.query().filter().first.return_value = type('Chat', (), mock_chat)
        mock_session.query().filter().order_by().all.return_value = [
            type('Message', (), mock_message)
        ]
        mock_db.return_value = mock_session

        response = client.get(f"/chats/{mock_chat['id']}")
        
        assert response.status_code == 200
        result = response.json()
        assert "chat" in result
        assert "messages" in result
        assert result["chat"]["id"] == mock_chat["id"]
        assert len(result["messages"]) == 1
        assert result["messages"][0]["id"] == mock_message["id"]


@pytest.mark.asyncio
async def test_get_chat_not_found():
    """Test getting a non-existent chat."""
    with patch('routes.chats.get_db') as mock_db:
        mock_session = MagicMock()
        mock_session.query().filter().first.return_value = None
        mock_db.return_value = mock_session

        response = client.get("/chats/non-existent-id")
        
        assert response.status_code == 404
        assert "Chat not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_delete_chat():
    """Test deleting a chat."""
    mock_chat = create_mock_chat()

    with patch('routes.chats.get_db') as mock_db:
        mock_session = MagicMock()
        mock_session.query().filter().first.return_value = type('Chat', (), mock_chat)
        mock_db.return_value = mock_session

        response = client.delete(f"/chats/{mock_chat['id']}")
        
        assert response.status_code == 200
        result = response.json()
        assert "message" in result
        assert "successfully" in result["message"].lower()
        
        # Verify delete operations were called
        mock_session.query().filter().delete.assert_called_once()
        mock_session.delete.assert_called_once()
        mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_chat_not_found():
    """Test deleting a non-existent chat."""
    with patch('routes.chats.get_db') as mock_db:
        mock_session = MagicMock()
        mock_session.query().filter().first.return_value = None
        mock_db.return_value = mock_session

        response = client.delete("/chats/non-existent-id")
        
        assert response.status_code == 404
        assert "Chat not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_database_error():
    """Test handling of database errors."""
    with patch('routes.chats.get_db') as mock_db:
        mock_session = MagicMock()
        mock_session.query.side_effect = Exception("Database error")
        mock_db.return_value = mock_session

        response = client.get("/chats")
        
        assert response.status_code == 500
        assert "internal server error" in response.json()["detail"].lower()
