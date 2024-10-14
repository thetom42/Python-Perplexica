import pytest
from unittest.mock import AsyncMock, patch
from langchain.schema import HumanMessage, AIMessage
from backend.agents.academic_search_agent import handle_academic_search

@pytest.fixture
def mock_chat_model():
    return AsyncMock()

@pytest.fixture
def mock_embeddings_model():
    return AsyncMock()

@pytest.mark.asyncio
async def test_handle_academic_search(mock_chat_model, mock_embeddings_model):
    query = "Recent advancements in quantum computing"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I help you with academic research today?")
    ]

    with patch('backend.agents.academic_search_agent.search_academic_sources') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {
            "results": [
                {"title": "Quantum Computing: Recent Advancements", "abstract": "This paper discusses the latest developments in quantum computing."}
            ]
        }

        mock_chat_model.arun.return_value = "Recent advancements in quantum computing include improvements in qubit stability and error correction techniques."

        result = await handle_academic_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "quantum computing" in result["data"].lower()
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_academic_search_no_results(mock_chat_model, mock_embeddings_model):
    query = "Non-existent research topic"
    history = []

    with patch('backend.agents.academic_search_agent.search_academic_sources') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {"results": []}

        mock_chat_model.arun.return_value = "I couldn't find any relevant academic papers on this topic."

        result = await handle_academic_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "couldn't find any relevant" in result["data"].lower()
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_academic_search_multiple_results(mock_chat_model, mock_embeddings_model):
    query = "Machine learning in healthcare"
    history = []

    with patch('backend.agents.academic_search_agent.search_academic_sources') as mock_search, \
         patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_search.return_value = {
            "results": [
                {"title": "ML in Diagnostics", "abstract": "Application of machine learning in medical diagnostics."},
                {"title": "AI-driven Drug Discovery", "abstract": "Using AI and ML techniques for drug discovery."}
            ]
        }

        mock_chat_model.arun.return_value = "Machine learning is being applied in healthcare for improved diagnostics and drug discovery."

        result = await handle_academic_search(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "machine learning" in result["data"].lower()
        assert "healthcare" in result["data"].lower()
        mock_search.assert_called_once_with(query)
        mock_chat_model.arun.assert_called_once()
