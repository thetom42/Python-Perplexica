import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain.schema import HumanMessage, AIMessage, Document
from backend.agents.reddit_search_agent import (
    handle_reddit_search,
    create_basic_reddit_search_retriever_chain,
    create_basic_reddit_search_answering_chain,
    RunnableSequence
)

@pytest.fixture
def mock_chat_model():
    mock = AsyncMock()
    mock.arun = AsyncMock()
    return mock

@pytest.fixture
def mock_embeddings_model():
    mock = AsyncMock()
    mock.embed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock.embed_query = AsyncMock(return_value=[0.2, 0.3, 0.4])
    return mock

@pytest.fixture
def sample_reddit_results():
    return {
        "results": [
            {
                "title": "Best Pizza NYC Thread",
                "url": "https://reddit.com/r/nyc/pizza",
                "content": "Joe's Pizza on Carmine Street is amazing"
            },
            {
                "title": "NYC Food Guide",
                "url": "https://reddit.com/r/food/nyc",
                "content": "Lombardi's is the oldest pizzeria"
            }
        ]
    }

@pytest.mark.asyncio
async def test_handle_reddit_search_basic(mock_chat_model, mock_embeddings_model):
    query = "Best pizza in New York"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!")
    ]

    with patch('backend.agents.reddit_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {
            "results": [
                {
                    "title": "NYC Pizza Discussion",
                    "url": "https://reddit.com/r/nyc/pizza",
                    "content": "Joe's Pizza is the best"
                }
            ]
        }
        mock_chat_model.arun.return_value = "According to Reddit, Joe's Pizza is highly recommended [1]"

        async for result in handle_reddit_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "response":
                assert "[1]" in result["data"]
                assert "Joe's Pizza" in result["data"]
            elif result["type"] == "sources":
                assert len(result["data"]) > 0
                assert "url" in result["data"][0]

@pytest.mark.asyncio
async def test_retriever_chain_creation(mock_chat_model):
    chain = await create_basic_reddit_search_retriever_chain(mock_chat_model)
    assert isinstance(chain, RunnableSequence)
    
    mock_chat_model.arun.return_value = "rephrased query"
    with patch('backend.agents.reddit_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        
        result = await chain.invoke({
            "query": "test query",
            "chat_history": []
        })
        
        assert "query" in result
        assert "docs" in result

@pytest.mark.asyncio
async def test_optimization_modes(mock_chat_model, mock_embeddings_model, sample_reddit_results):
    query = "test query"
    history = []
    
    with patch('backend.agents.reddit_search_agent.search_searxng') as mock_search:
        mock_search.return_value = sample_reddit_results
        
        for mode in ["speed", "balanced", "quality"]:
            async for result in handle_reddit_search(
                query, history, mock_chat_model, mock_embeddings_model, mode
            ):
                if result["type"] == "sources":
                    assert isinstance(result["data"], list)
                elif result["type"] == "response":
                    assert isinstance(result["data"], str)

@pytest.mark.asyncio
async def test_document_reranking(mock_chat_model, mock_embeddings_model):
    query = "test query"
    history = []
    
    with patch('backend.agents.reddit_search_agent.search_searxng') as mock_search:
        # Create test documents with varying relevance
        mock_search.return_value = {
            "results": [{"title": f"Result {i}", "url": f"url{i}", "content": f"content {i}"} for i in range(20)]
        }
        
        async for result in handle_reddit_search(
            query, history, mock_chat_model, mock_embeddings_model, "balanced"
        ):
            if result["type"] == "sources":
                # Should be limited and reranked
                assert len(result["data"]) <= 15

@pytest.mark.asyncio
async def test_error_handling(mock_chat_model, mock_embeddings_model):
    query = "test query"
    history = []

    with patch('backend.agents.reddit_search_agent.search_searxng') as mock_search:
        mock_search.side_effect = Exception("Search API error")
        
        async for result in handle_reddit_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "error":
                assert "error" in result["data"].lower()

@pytest.mark.asyncio
async def test_empty_query_handling(mock_chat_model, mock_embeddings_model):
    query = ""
    history = []

    async for result in handle_reddit_search(query, history, mock_chat_model, mock_embeddings_model):
        if result["type"] == "response":
            assert "specific question" in result["data"].lower()

@pytest.mark.asyncio
async def test_query_rephrasing(mock_chat_model, mock_embeddings_model):
    query = "What do Redditors think about Python?"
    history = [
        HumanMessage(content="I'm interested in programming"),
        AIMessage(content="I can help with that!")
    ]

    with patch('backend.agents.reddit_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        mock_chat_model.arun.return_value = "Python programming language opinions"

        async for _ in handle_reddit_search(query, history, mock_chat_model, mock_embeddings_model):
            pass
        
        # Verify the query was processed through the rephrasing chain
        assert mock_chat_model.arun.called
        call_args = mock_chat_model.arun.call_args[1]
        assert "chat_history" in call_args
        assert "query" in call_args

@pytest.mark.asyncio
async def test_source_tracking(mock_chat_model, mock_embeddings_model, sample_reddit_results):
    query = "test query"
    history = []

    with patch('backend.agents.reddit_search_agent.search_searxng') as mock_search:
        mock_search.return_value = sample_reddit_results
        mock_chat_model.arun.return_value = "Response with citation [1][2]"

        sources_received = False
        response_received = False
        
        async for result in handle_reddit_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "sources":
                sources_received = True
                assert len(result["data"]) == len(sample_reddit_results["results"])
                for source in result["data"]:
                    assert "title" in source
                    assert "url" in source
            elif result["type"] == "response":
                response_received = True
                assert "[1]" in result["data"]
                assert "[2]" in result["data"]
        
        assert sources_received and response_received

@pytest.mark.asyncio
async def test_invalid_history_handling(mock_chat_model, mock_embeddings_model):
    query = "test query"
    invalid_history = ["not a valid message"]  # Invalid history format

    with patch('backend.agents.reddit_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        
        async for result in handle_reddit_search(query, invalid_history, mock_chat_model, mock_embeddings_model):
            # Should handle invalid history gracefully
            assert result["type"] in ["response", "sources", "error"]

@pytest.mark.asyncio
async def test_malformed_search_results(mock_chat_model, mock_embeddings_model):
    query = "test query"
    history = []

    with patch('backend.agents.reddit_search_agent.search_searxng') as mock_search:
        # Missing required fields in results
        mock_search.return_value = {
            "results": [
                {"title": "Test"},  # Missing url and content
                {"url": "https://reddit.com"},  # Missing title and content
                {
                    "title": "Complete",
                    "url": "https://reddit.com/complete",
                    "content": "Complete content"
                }
            ]
        }
        
        async for result in handle_reddit_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "sources":
                # Should only include the complete result
                assert len(result["data"]) == 1
                assert result["data"][0]["title"] == "Complete"
