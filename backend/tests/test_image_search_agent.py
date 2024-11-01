"""
Tests for the image search agent functionality.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain.schema import HumanMessage, AIMessage
from agents.image_search_agent import ImageSearchAgent, handle_image_search

@pytest.fixture
def mock_chat_model():
    """Create a mock chat model."""
    mock = AsyncMock()
    mock.invoke = AsyncMock()
    return mock

@pytest.fixture
def mock_embeddings_model():
    """Create a mock embeddings model."""
    return AsyncMock()

@pytest.fixture
def sample_search_result():
    """Create a sample search result."""
    return {
        "results": [
            {
                "title": "Eiffel Tower",
                "url": "https://example.com/eiffel1.jpg",
                "img_src": "https://example.com/eiffel1_thumb.jpg"
            },
            {
                "title": "Eiffel Tower Night",
                "url": "https://example.com/eiffel2.jpg",
                "img_src": "https://example.com/eiffel2_thumb.jpg"
            }
        ]
    }

@pytest.mark.asyncio
async def test_image_search_agent_basic(mock_chat_model, mock_embeddings_model, sample_search_result):
    """Test basic image search functionality."""
    agent = ImageSearchAgent(mock_chat_model, mock_embeddings_model)

    with patch('agents.image_search_agent.search_searxng') as mock_search:
        mock_search.return_value = sample_search_result
        mock_chat_model.invoke.return_value.content = "Eiffel Tower"

        retriever_chain = await agent.create_retriever_chain()
        result = await retriever_chain.invoke({
            "query": "Show me the Eiffel Tower",
            "chat_history": []
        })

        assert "query" in result
        assert "images" in result
        assert len(result["images"]) == 2
        assert result["images"][0]["title"] == "Eiffel Tower"

@pytest.mark.asyncio
async def test_image_search_agent_no_results(mock_chat_model, mock_embeddings_model):
    """Test handling of no search results."""
    agent = ImageSearchAgent(mock_chat_model, mock_embeddings_model)

    with patch('agents.image_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        mock_chat_model.invoke.return_value.content = "test query"

        answering_chain = await agent.create_answering_chain()
        result = await answering_chain.invoke({
            "query": "nonexistent image",
            "chat_history": []
        })

        assert result["type"] == "error"
        assert "No images found" in result["data"]

@pytest.mark.asyncio
async def test_image_search_agent_error_handling(mock_chat_model, mock_embeddings_model):
    """Test error handling in image search."""
    agent = ImageSearchAgent(mock_chat_model, mock_embeddings_model)

    with patch('agents.image_search_agent.search_searxng') as mock_search:
        mock_search.side_effect = Exception("Search API error")
        mock_chat_model.invoke.return_value.content = "test query"

        answering_chain = await agent.create_answering_chain()
        result = await answering_chain.invoke({
            "query": "test query",
            "chat_history": []
        })

        assert result["type"] == "error"
        assert "error occurred" in result["data"].lower()

@pytest.mark.asyncio
async def test_image_search_agent_malformed_results(mock_chat_model, mock_embeddings_model):
    """Test handling of malformed search results."""
    agent = ImageSearchAgent(mock_chat_model, mock_embeddings_model)

    with patch('agents.image_search_agent.search_searxng') as mock_search:
        # Missing required fields in results
        mock_search.return_value = {
            "results": [
                {"title": "Test"},  # Missing url and img_src
                {"url": "https://example.com"},  # Missing title and img_src
                {
                    "title": "Complete",
                    "url": "https://example.com/complete",
                    "img_src": "https://example.com/complete.jpg"
                }
            ]
        }
        mock_chat_model.invoke.return_value.content = "test query"

        retriever_chain = await agent.create_retriever_chain()
        result = await retriever_chain.invoke({
            "query": "test query",
            "chat_history": []
        })

        assert len(result["images"]) == 1
        assert result["images"][0]["title"] == "Complete"

@pytest.mark.asyncio
async def test_image_search_agent_result_limit(mock_chat_model, mock_embeddings_model, sample_search_result):
    """Test limiting of search results."""
    agent = ImageSearchAgent(mock_chat_model, mock_embeddings_model)

    with patch('agents.image_search_agent.search_searxng') as mock_search:
        # Create more than 10 results
        results = {"results": sample_search_result["results"] * 6}  # 12 results
        mock_search.return_value = results
        mock_chat_model.invoke.return_value.content = "test query"

        retriever_chain = await agent.create_retriever_chain()
        result = await retriever_chain.invoke({
            "query": "test query",
            "chat_history": []
        })

        assert len(result["images"]) <= 10

@pytest.mark.asyncio
async def test_handle_image_search_integration(mock_chat_model, mock_embeddings_model, sample_search_result):
    """Test the handle_image_search function integration."""
    with patch('agents.image_search_agent.search_searxng') as mock_search:
        mock_search.return_value = sample_search_result
        mock_chat_model.invoke.return_value.content = "test query"

        results = []
        async for result in handle_image_search(
            "test query",
            [],
            mock_chat_model,
            mock_embeddings_model
        ):
            results.append(result)

        assert len(results) == 1
        assert results[0]["type"] == "images"
        assert len(results[0]["data"]) == 2

@pytest.mark.asyncio
async def test_multiple_search_engines_integration(mock_chat_model, mock_embeddings_model):
    """Test integration with multiple search engines."""
    agent = ImageSearchAgent(mock_chat_model, mock_embeddings_model)

    with patch('agents.image_search_agent.search_searxng') as mock_search:
        mock_chat_model.invoke.return_value.content = "test query"
        await agent.create_retriever_chain()

        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert "engines" in call_kwargs
        assert "bing images" in call_kwargs["engines"]
        assert "google images" in call_kwargs["engines"]
