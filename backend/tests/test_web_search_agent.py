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

@pytest.mark.asyncio
async def test_handle_web_search_multiple_results(mock_chat_model, mock_embeddings_model):
    query = "What are the largest cities in Japan?"
    history = []

    with patch('backend.agents.web_search_agent.search_searxng') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {
            "results": [
                {"content": "Tokyo is the largest city in Japan."},
                {"content": "Yokohama is the second largest city in Japan."},
                {"content": "Osaka is the third largest city in Japan."}
            ]
        }

        mock_chat_model.arun.return_value = "The largest cities in Japan are Tokyo, Yokohama, and Osaka, in that order."

        result = await handle_web_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "Tokyo" in result["data"]
        assert "Yokohama" in result["data"]
        assert "Osaka" in result["data"]
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_web_search_error_handling(mock_chat_model, mock_embeddings_model):
    query = "What is the weather like today?"
    history = []

    with patch('backend.agents.web_search_agent.search_searxng') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.side_effect = Exception("Search engine error")

        result = await handle_web_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "error"
        assert "An error occurred while searching" in result["data"]
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_not_called()

@pytest.mark.asyncio
async def test_handle_web_search_with_history(mock_chat_model, mock_embeddings_model):
    query = "What's its population?"
    history = [
        HumanMessage(content="What's the capital of Japan?"),
        AIMessage(content="The capital of Japan is Tokyo.")
    ]

    with patch('backend.agents.web_search_agent.search_searxng') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {
            "results": [
                {"content": "The population of Tokyo is approximately 13.96 million."}
            ]
        }

        mock_chat_model.arun.return_value = "The population of Tokyo is approximately 13.96 million."

        result = await handle_web_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "13.96 million" in result["data"]
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()
        
        # Check if the history was passed to the chat model
        call_args = mock_chat_model.arun.call_args[0][0]
        assert any("capital of Japan is Tokyo" in str(msg.content) for msg in call_args)
        assert any("What's its population?" in str(msg.content) for msg in call_args)
