import pytest
from unittest.mock import AsyncMock, patch
from langchain.schema import HumanMessage, AIMessage
from backend.agents.youtube_search_agent import handle_youtube_search

@pytest.fixture
def mock_chat_model():
    return AsyncMock()

@pytest.fixture
def mock_embeddings_model():
    return AsyncMock()

@pytest.mark.asyncio
async def test_handle_youtube_search(mock_chat_model, mock_embeddings_model):
    query = "Best guitar tutorials for beginners"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I help you find YouTube videos today?")
    ]

    with patch('backend.agents.youtube_search_agent.search_youtube') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {
            "results": [
                {"title": "Complete Guitar Course for Beginners", "description": "Learn guitar basics from scratch with this comprehensive tutorial."}
            ]
        }

        mock_chat_model.arun.return_value = "I found a great guitar tutorial for beginners on YouTube titled 'Complete Guitar Course for Beginners'. This video provides a comprehensive tutorial that teaches guitar basics from scratch, perfect for those just starting out."

        result = await handle_youtube_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "Complete Guitar Course for Beginners" in result["data"]
        assert "comprehensive tutorial" in result["data"].lower()
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_youtube_search_no_results(mock_chat_model, mock_embeddings_model):
    query = "Non-existent video topic on YouTube"
    history = []

    with patch('backend.agents.youtube_search_agent.search_youtube') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {"results": []}

        mock_chat_model.arun.return_value = "I'm sorry, but I couldn't find any relevant YouTube videos on that topic. Could you try rephrasing your query or searching for a related topic?"

        result = await handle_youtube_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "couldn't find any relevant YouTube videos" in result["data"].lower()
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_youtube_search_multiple_results(mock_chat_model, mock_embeddings_model):
    query = "Python programming tutorials"
    history = []

    with patch('backend.agents.youtube_search_agent.search_youtube') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {
            "results": [
                {"title": "Python for Beginners - Full Course", "description": "Learn Python basics in this comprehensive course for newcomers to programming."},
                {"title": "Advanced Python Techniques", "description": "Discover advanced Python concepts and best practices for experienced programmers."}
            ]
        }

        mock_chat_model.arun.return_value = "I found two relevant Python programming tutorials on YouTube:\n1. 'Python for Beginners - Full Course': This comprehensive course covers Python basics and is perfect for newcomers to programming.\n2. 'Advanced Python Techniques': This video focuses on advanced Python concepts and best practices, suitable for experienced programmers looking to enhance their skills."

        result = await handle_youtube_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "Python for Beginners" in result["data"]
        assert "Advanced Python Techniques" in result["data"]
        assert "newcomers" in result["data"].lower()
        assert "experienced programmers" in result["data"].lower()
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()
