import pytest
from unittest.mock import AsyncMock, patch
from langchain.schema import HumanMessage, AIMessage
from backend.agents.video_search_agent import handle_video_search

@pytest.fixture
def mock_chat_model():
    return AsyncMock()

@pytest.fixture
def mock_embeddings_model():
    return AsyncMock()

@pytest.mark.asyncio
async def test_handle_video_search(mock_chat_model, mock_embeddings_model):
    query = "Best cooking tutorials for beginners"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I help you find videos today?")
    ]

    with patch('backend.agents.video_search_agent.search_videos') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {
            "results": [
                {"title": "Cooking Basics for Beginners", "description": "Learn essential cooking techniques for novice chefs."}
            ]
        }

        mock_chat_model.arun.return_value = "I found a great cooking tutorial for beginners titled 'Cooking Basics for Beginners'. This video covers essential cooking techniques that are perfect for novice chefs."

        result = await handle_video_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "Cooking Basics for Beginners" in result["data"]
        assert "essential cooking techniques" in result["data"].lower()
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_video_search_no_results(mock_chat_model, mock_embeddings_model):
    query = "Non-existent video topic"
    history = []

    with patch('backend.agents.video_search_agent.search_videos') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {"results": []}

        mock_chat_model.arun.return_value = "I'm sorry, but I couldn't find any relevant videos on that topic."

        result = await handle_video_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "couldn't find any relevant videos" in result["data"].lower()
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_video_search_multiple_results(mock_chat_model, mock_embeddings_model):
    query = "Python programming tutorials"
    history = []

    with patch('backend.agents.video_search_agent.search_videos') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {
            "results": [
                {"title": "Python for Beginners", "description": "Complete Python tutorial for newcomers to programming."},
                {"title": "Advanced Python Techniques", "description": "Learn advanced Python concepts and best practices."}
            ]
        }

        mock_chat_model.arun.return_value = "I found two relevant Python programming tutorials. The first one, 'Python for Beginners', is a complete tutorial for newcomers to programming. The second one, 'Advanced Python Techniques', covers advanced concepts and best practices for more experienced programmers."

        result = await handle_video_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "Python for Beginners" in result["data"]
        assert "Advanced Python Techniques" in result["data"]
        assert "newcomers" in result["data"].lower()
        assert "advanced concepts" in result["data"].lower()
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()
