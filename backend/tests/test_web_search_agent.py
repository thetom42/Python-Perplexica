import pytest
from unittest.mock import AsyncMock, patch
from langchain.schema import HumanMessage, AIMessage
from backend.agents.web_search_agent import handle_web_search
from backend.lib.providers import get_chat_model, get_embeddings_model

@pytest.fixture
def mock_chat_model():
    return AsyncMock()

@pytest.fixture
def mock_embeddings_model():
    return AsyncMock()

@pytest.mark.asyncio
async def test_handle_web_search(mock_chat_model, mock_embeddings_model):
    query = "What is the capital of France?"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I help you today?")
    ]

    with patch('backend.agents.web_search_agent.search_searxng') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {
            "results": [
                {"content": "Paris is the capital of France."}
            ]
        }

        mock_chat_model.arun.return_value = "The capital of France is Paris."

        result = await handle_web_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "Paris" in result["data"]
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_web_search_no_results(mock_chat_model, mock_embeddings_model):
    query = "What is the capital of Atlantis?"
    history = []

    with patch('backend.agents.web_search_agent.search_searxng') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {"results": []}

        mock_chat_model.arun.return_value = "I couldn't find any relevant information on this topic."

        result = await handle_web_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "couldn't find any relevant information" in result["data"]
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()
