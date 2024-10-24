import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain.schema import HumanMessage, AIMessage, Document
from backend.agents.wolfram_alpha_search_agent import (
    handle_wolfram_alpha_search,
    create_basic_wolfram_alpha_search_retriever_chain,
    create_basic_wolfram_alpha_search_answering_chain,
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
def sample_wolfram_results():
    return {
        "results": [
            {
                "title": "Population Query",
                "url": "https://www.wolframalpha.com/population",
                "content": "Tokyo population: 13.96 million (2023)"
            },
            {
                "title": "Additional Info",
                "url": "https://www.wolframalpha.com/tokyo",
                "content": "Metropolitan area: 37.4 million"
            }
        ]
    }

@pytest.mark.asyncio
async def test_handle_wolfram_alpha_search_basic(mock_chat_model, mock_embeddings_model):
    query = "What is the population of Tokyo?"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!")
    ]

    with patch('backend.agents.wolfram_alpha_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {
            "results": [
                {
                    "title": "Tokyo Population",
                    "url": "https://www.wolframalpha.com/result",
                    "content": "13.96 million people (2023)"
                }
            ]
        }
        mock_chat_model.arun.return_value = "The population of Tokyo is 13.96 million people (2023) [1]"

        async for result in handle_wolfram_alpha_search(query, history, mock_chat_model):
            if result["type"] == "response":
                assert "[1]" in result["data"]
                assert "13.96 million" in result["data"]
            elif result["type"] == "sources":
                assert len(result["data"]) > 0
                assert "url" in result["data"][0]

@pytest.mark.asyncio
async def test_retriever_chain_creation(mock_chat_model):
    chain = await create_basic_wolfram_alpha_search_retriever_chain(mock_chat_model)
    assert isinstance(chain, RunnableSequence)
    
    with patch('backend.agents.wolfram_alpha_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        mock_chat_model.arun.return_value = "rephrased query"
        
        result = await chain.invoke({
            "query": "test query",
            "chat_history": []
        })
        
        assert "query" in result
        assert "docs" in result

@pytest.mark.asyncio
async def test_query_rephrasing(mock_chat_model):
    query = "What's the derivative of x^2?"
    history = [
        HumanMessage(content="I need help with calculus"),
        AIMessage(content="I can help with that!")
    ]

    with patch('backend.agents.wolfram_alpha_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        mock_chat_model.arun.return_value = "derivative of x^2"

        async for _ in handle_wolfram_alpha_search(query, history, mock_chat_model):
            pass
        
        # Verify the query was processed through the rephrasing chain
        assert mock_chat_model.arun.called
        call_args = mock_chat_model.arun.call_args[1]
        assert "chat_history" in call_args
        assert "query" in call_args

@pytest.mark.asyncio
async def test_wolfram_alpha_specific_search(mock_chat_model):
    query = "Calculate sin(30)"
    history = []

    with patch('backend.agents.wolfram_alpha_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        
        async for _ in handle_wolfram_alpha_search(query, history, mock_chat_model):
            pass
        
        # Verify Wolfram Alpha specific search configuration
        mock_search.assert_called_once()
        call_args = mock_search.call_args[1]
        assert call_args.get("engines") == ["wolframalpha"]

@pytest.mark.asyncio
async def test_error_handling(mock_chat_model):
    query = "test query"
    history = []

    with patch('backend.agents.wolfram_alpha_search_agent.search_searxng') as mock_search:
        mock_search.side_effect = Exception("Search API error")
        
        async for result in handle_wolfram_alpha_search(query, history, mock_chat_model):
            if result["type"] == "error":
                assert "error" in result["data"].lower()

@pytest.mark.asyncio
async def test_empty_query_handling(mock_chat_model):
    query = ""
    history = []

    async for result in handle_wolfram_alpha_search(query, history, mock_chat_model):
        if result["type"] == "response":
            assert "specific question" in result["data"].lower()

@pytest.mark.asyncio
async def test_source_tracking(mock_chat_model, sample_wolfram_results):
    query = "test query"
    history = []

    with patch('backend.agents.wolfram_alpha_search_agent.search_searxng') as mock_search:
        mock_search.return_value = sample_wolfram_results
        mock_chat_model.arun.return_value = "Response with citation [1][2]"

        sources_received = False
        response_received = False
        
        async for result in handle_wolfram_alpha_search(query, history, mock_chat_model):
            if result["type"] == "sources":
                sources_received = True
                assert len(result["data"]) == len(sample_wolfram_results["results"])
                for source in result["data"]:
                    assert "title" in source
                    assert "url" in source
            elif result["type"] == "response":
                response_received = True
                assert "[1]" in result["data"]
                assert "[2]" in result["data"]
        
        assert sources_received and response_received

@pytest.mark.asyncio
async def test_unused_embeddings_parameter(mock_chat_model, mock_embeddings_model):
    query = "test query"
    history = []

    with patch('backend.agents.wolfram_alpha_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        
        # Should work the same with or without embeddings
        async for result in handle_wolfram_alpha_search(query, history, mock_chat_model, mock_embeddings_model):
            assert result["type"] in ["response", "sources", "error"]
        
        async for result in handle_wolfram_alpha_search(query, history, mock_chat_model, None):
            assert result["type"] in ["response", "sources", "error"]

@pytest.mark.asyncio
async def test_complex_mathematical_query(mock_chat_model):
    query = "Solve x^2 + 2x + 1 = 0"
    history = []

    with patch('backend.agents.wolfram_alpha_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {
            "results": [
                {
                    "title": "Quadratic Equation",
                    "url": "https://www.wolframalpha.com/solve",
                    "content": "x = -1 (double root)"
                }
            ]
        }
        mock_chat_model.arun.return_value = "The solution to x^2 + 2x + 1 = 0 is x = -1 (double root) [1]"

        async for result in handle_wolfram_alpha_search(query, history, mock_chat_model):
            if result["type"] == "response":
                assert "x = -1" in result["data"]
                assert "[1]" in result["data"]

@pytest.mark.asyncio
async def test_malformed_search_results(mock_chat_model):
    query = "test query"
    history = []

    with patch('backend.agents.wolfram_alpha_search_agent.search_searxng') as mock_search:
        # Missing required fields in results
        mock_search.return_value = {
            "results": [
                {"title": "Test"},  # Missing url and content
                {"url": "https://example.com"},  # Missing title and content
                {
                    "title": "Complete",
                    "url": "https://example.com/complete",
                    "content": "Complete content"
                }
            ]
        }
        
        async for result in handle_wolfram_alpha_search(query, history, mock_chat_model):
            if result["type"] == "sources":
                # Should handle malformed results gracefully
                assert len(result["data"]) > 0
