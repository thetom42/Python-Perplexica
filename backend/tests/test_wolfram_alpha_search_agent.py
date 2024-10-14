import pytest
from unittest.mock import AsyncMock, patch
from langchain.schema import HumanMessage, AIMessage
from backend.agents.wolfram_alpha_search_agent import handle_wolfram_alpha_search

@pytest.fixture
def mock_chat_model():
    return AsyncMock()

@pytest.fixture
def mock_embeddings_model():
    return AsyncMock()

@pytest.mark.asyncio
async def test_handle_wolfram_alpha_search(mock_chat_model, mock_embeddings_model):
    query = "What is the population of Tokyo?"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I help you with factual information today?")
    ]

    with patch('backend.agents.wolfram_alpha_search_agent.search_wolfram_alpha') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {
            "results": [
                {"title": "Population of Tokyo", "content": "13.96 million people (2023 estimate)"}
            ]
        }

        mock_chat_model.arun.return_value = "According to the latest estimate from Wolfram Alpha, the population of Tokyo is approximately 13.96 million people as of 2023."

        result = await handle_wolfram_alpha_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "13.96 million" in result["data"]
        assert "2023" in result["data"]
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_wolfram_alpha_search_no_results(mock_chat_model, mock_embeddings_model):
    query = "What is the population of Atlantis?"
    history = []

    with patch('backend.agents.wolfram_alpha_search_agent.search_wolfram_alpha') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {"results": []}

        mock_chat_model.arun.return_value = "I'm sorry, but I couldn't find any factual information about the population of Atlantis. Atlantis is a mythical city and doesn't have a real population."

        result = await handle_wolfram_alpha_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "couldn't find any factual information" in result["data"].lower()
        assert "mythical city" in result["data"].lower()
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_wolfram_alpha_search_complex_query(mock_chat_model, mock_embeddings_model):
    query = "What is the derivative of x^2 + 3x + 2?"
    history = []

    with patch('backend.agents.wolfram_alpha_search_agent.search_wolfram_alpha') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {
            "results": [
                {"title": "Derivative", "content": "d/dx(x^2 + 3x + 2) = 2x + 3"},
                {"title": "Step-by-step solution", "content": "1. d/dx(x^2) = 2x\n2. d/dx(3x) = 3\n3. d/dx(2) = 0\n4. Sum the results: 2x + 3 + 0 = 2x + 3"}
            ]
        }

        mock_chat_model.arun.return_value = "The derivative of x^2 + 3x + 2 is 2x + 3. Here's a step-by-step solution:\n1. The derivative of x^2 is 2x\n2. The derivative of 3x is 3\n3. The derivative of 2 (constant) is 0\n4. Sum all the terms: 2x + 3 + 0 = 2x + 3"

        result = await handle_wolfram_alpha_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "2x + 3" in result["data"]
        assert "step-by-step" in result["data"].lower()
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()
