import pytest
from unittest.mock import AsyncMock, patch
from langchain.schema import HumanMessage, AIMessage
from backend.agents.suggestion_generator_agent import handle_suggestion_generation

@pytest.fixture
def mock_chat_model():
    return AsyncMock()

@pytest.fixture
def mock_embeddings_model():
    return AsyncMock()

@pytest.mark.asyncio
async def test_handle_suggestion_generation(mock_chat_model, mock_embeddings_model):
    query = "Suggest some healthy breakfast ideas"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I help you with suggestions today?")
    ]

    with patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_chat_model.arun.return_value = "Here are some healthy breakfast ideas:\n1. Oatmeal with fresh berries and nuts\n2. Greek yogurt parfait with granola and honey\n3. Whole grain toast with avocado and poached eggs"

        result = await handle_suggestion_generation(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "oatmeal" in result["data"].lower()
        assert "greek yogurt" in result["data"].lower()
        assert "whole grain toast" in result["data"].lower()
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_suggestion_generation_empty_query(mock_chat_model, mock_embeddings_model):
    query = ""
    history = []

    with patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_chat_model.arun.return_value = "I'm sorry, but I need more information to provide suggestions. Could you please specify what kind of suggestions you're looking for?"

        result = await handle_suggestion_generation(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "need more information" in result["data"].lower()
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_suggestion_generation_with_context(mock_chat_model, mock_embeddings_model):
    query = "Suggest some team-building activities"
    history = [
        HumanMessage(content="I'm planning a corporate retreat"),
        AIMessage(content="That sounds exciting! How can I assist you with planning your corporate retreat?")
    ]

    with patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_chat_model.arun.return_value = "Here are some team-building activities suitable for a corporate retreat:\n1. Escape room challenge\n2. Group cooking class\n3. Outdoor scavenger hunt\n4. Improv workshop\n5. Volunteer project in the local community"

        result = await handle_suggestion_generation(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "escape room" in result["data"].lower()
        assert "cooking class" in result["data"].lower()
        assert "scavenger hunt" in result["data"].lower()
        mock_chat_model.arun.assert_called_once()
