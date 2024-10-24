import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain.schema import HumanMessage, AIMessage
from backend.agents.image_search_agent import (
    handle_image_search,
    create_image_search_chain,
    RunnableSequence
)

@pytest.fixture
def mock_chat_model():
    mock = AsyncMock()
    mock.arun = AsyncMock()
    return mock

@pytest.fixture
def mock_embeddings_model():
    return AsyncMock()

@pytest.fixture
def sample_search_result():
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
async def test_handle_image_search_basic(mock_chat_model):
    query = "Show me images of the Eiffel Tower"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I help you today?")
    ]

    with patch('backend.agents.image_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {
            "results": [
                {
                    "title": "Eiffel Tower",
                    "url": "https://example.com/eiffel.jpg",
                    "img_src": "https://example.com/thumb.jpg"
                }
            ]
        }
        mock_chat_model.arun.return_value = "Eiffel Tower"

        results = await handle_image_search(query, history, mock_chat_model)
        
        assert len(results) == 1
        assert results[0]["title"] == "Eiffel Tower"
        assert "img_src" in results[0]
        assert "url" in results[0]
        mock_search.assert_called_once()

@pytest.mark.asyncio
async def test_handle_image_search_no_results(mock_chat_model):
    query = "Show me images of a non-existent object"
    history = []

    with patch('backend.agents.image_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        mock_chat_model.arun.return_value = "non-existent object"

        results = await handle_image_search(query, history, mock_chat_model)
        
        assert len(results) == 0
        mock_search.assert_called_once()

@pytest.mark.asyncio
async def test_create_image_search_chain(mock_chat_model):
    chain = await create_image_search_chain(mock_chat_model)
    assert isinstance(chain, RunnableSequence)
    
    with patch('backend.agents.image_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        mock_chat_model.arun.return_value = "test query"
        
        result = await chain.invoke({
            "query": "test query",
            "chat_history": []
        })
        
        assert isinstance(result, list)

@pytest.mark.asyncio
async def test_handle_image_search_error_handling(mock_chat_model):
    query = "test query"
    history = []

    with patch('backend.agents.image_search_agent.search_searxng') as mock_search:
        mock_search.side_effect = Exception("Search API error")
        
        results = await handle_image_search(query, history, mock_chat_model)
        assert len(results) == 0

@pytest.mark.asyncio
async def test_handle_image_search_malformed_results(mock_chat_model):
    query = "test query"
    history = []

    with patch('backend.agents.image_search_agent.search_searxng') as mock_search:
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
        mock_chat_model.arun.return_value = "test query"

        results = await handle_image_search(query, history, mock_chat_model)
        
        # Should only include the complete result
        assert len(results) == 1
        assert results[0]["title"] == "Complete"

@pytest.mark.asyncio
async def test_handle_image_search_result_limit(mock_chat_model, sample_search_result):
    query = "test query"
    history = []

    with patch('backend.agents.image_search_agent.search_searxng') as mock_search:
        # Create more than 10 results
        results = sample_search_result["results"] * 6  # 12 results
        mock_search.return_value = {"results": results}
        mock_chat_model.arun.return_value = "test query"

        results = await handle_image_search(query, history, mock_chat_model)
        
        # Should be limited to 10 results
        assert len(results) <= 10

@pytest.mark.asyncio
async def test_query_rephrasing(mock_chat_model):
    query = "What does the Eiffel Tower look like at night?"
    history = [
        HumanMessage(content="I'm interested in Paris landmarks"),
        AIMessage(content="I can help you with that!")
    ]

    with patch('backend.agents.image_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        mock_chat_model.arun.return_value = "Eiffel Tower night"

        await handle_image_search(query, history, mock_chat_model)
        
        # Verify the rephrased query was used
        mock_chat_model.arun.assert_called_once()
        call_args = mock_chat_model.arun.call_args[1]
        assert "chat_history" in call_args
        assert "query" in call_args
        assert query in call_args["query"]

@pytest.mark.asyncio
async def test_multiple_search_engines_integration(mock_chat_model):
    query = "test query"
    history = []

    with patch('backend.agents.image_search_agent.search_searxng') as mock_search:
        await handle_image_search(query, history, mock_chat_model)
        
        # Verify search was called with multiple engines
        mock_search.assert_called_once()
        call_args = mock_search.call_args[1]
        assert "engines" in call_args
        engines = call_args["engines"]
        assert "bing images" in engines
        assert "google images" in engines

@pytest.mark.asyncio
async def test_empty_query_handling(mock_chat_model):
    query = ""
    history = []

    results = await handle_image_search(query, history, mock_chat_model)
    assert len(results) == 0

@pytest.mark.asyncio
async def test_invalid_history_handling(mock_chat_model):
    query = "test query"
    invalid_history = ["not a valid message"]  # Invalid history format

    with patch('backend.agents.image_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        
        # Should handle invalid history gracefully
        results = await handle_image_search(query, invalid_history, mock_chat_model)
        assert isinstance(results, list)
