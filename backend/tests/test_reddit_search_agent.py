import pytest
from unittest.mock import AsyncMock, patch
from langchain.schema import HumanMessage, AIMessage
from backend.agents.reddit_search_agent import handle_reddit_search

@pytest.fixture
def mock_chat_model():
    return AsyncMock()

@pytest.fixture
def mock_embeddings_model():
    return AsyncMock()

@pytest.mark.asyncio
async def test_handle_reddit_search(mock_chat_model, mock_embeddings_model):
    query = "Best pizza in New York"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I help you with Reddit information today?")
    ]

    with patch('backend.agents.reddit_search_agent.search_reddit') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {
            "results": [
                {"title": "NYC Pizza Recommendations", "content": "Joe's Pizza on Carmine Street is often considered one of the best."}
            ]
        }

        mock_chat_model.arun.return_value = "According to Reddit users, Joe's Pizza on Carmine Street is often recommended as one of the best pizzas in New York."

        result = await handle_reddit_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "Joe's Pizza" in result["data"]
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_reddit_search_no_results(mock_chat_model, mock_embeddings_model):
    query = "Non-existent topic on Reddit"
    history = []

    with patch('backend.agents.reddit_search_agent.search_reddit') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {"results": []}

        mock_chat_model.arun.return_value = "I couldn't find any relevant Reddit posts on this topic."

        result = await handle_reddit_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "couldn't find any relevant" in result["data"].lower()
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_reddit_search_multiple_results(mock_chat_model, mock_embeddings_model):
    query = "Popular video games"
    history = []

    with patch('backend.agents.reddit_search_agent.search_reddit') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {
            "results": [
                {"title": "Top Games of 2023", "content": "Elden Ring and Baldur's Gate 3 are highly popular."},
                {"title": "Indie Game Recommendations", "content": "Hades and Stardew Valley are beloved indie titles."}
            ]
        }

        mock_chat_model.arun.return_value = "According to Reddit discussions, popular video games include AAA titles like Elden Ring and Baldur's Gate 3, as well as indie games such as Hades and Stardew Valley."

        result = await handle_reddit_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "Elden Ring" in result["data"]
        assert "indie games" in result["data"].lower()
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()
