import pytest
from unittest.mock import AsyncMock, patch
from langchain.schema import HumanMessage, AIMessage
from backend.agents.image_search_agent import handle_image_search
from backend.lib.providers import get_chat_model, get_embeddings_model

@pytest.fixture
def mock_chat_model():
    return AsyncMock()

@pytest.fixture
def mock_embeddings_model():
    return AsyncMock()

@pytest.mark.asyncio
async def test_handle_image_search(mock_chat_model, mock_embeddings_model):
    query = "Show me images of the Eiffel Tower"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I help you today?")
    ]

    with patch('backend.agents.image_search_agent.search_searxng') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {
            "results": [
                {
                    "title": "Eiffel Tower",
                    "url": "https://example.com/eiffel_tower.jpg",
                    "thumbnail": "https://example.com/eiffel_tower_thumb.jpg"
                }
            ]
        }

        mock_chat_model.arun.return_value = "Here are some images of the Eiffel Tower in Paris, France."

        result = await handle_image_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "Eiffel Tower" in result["data"]
        assert "images" in result
        assert len(result["images"]) == 1
        assert result["images"][0]["title"] == "Eiffel Tower"
        mock_search.assert_called_once_with(query, {"engines": "google_images"})
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_image_search_no_results(mock_chat_model, mock_embeddings_model):
    query = "Show me images of a unicorn riding a bicycle"
    history = []

    with patch('backend.agents.image_search_agent.search_searxng') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {"results": []}

        mock_chat_model.arun.return_value = "I'm sorry, but I couldn't find any images matching your query."

        result = await handle_image_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "couldn't find any images" in result["data"]
        assert "images" in result
        assert len(result["images"]) == 0
        mock_search.assert_called_once_with(query, {"engines": "google_images"})
        mock_chat_model.arun.assert_called_once()
