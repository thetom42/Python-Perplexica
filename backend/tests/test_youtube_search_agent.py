import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain.schema import HumanMessage, AIMessage, Document
from agents.youtube_search_agent import (
    handle_youtube_search,
    create_basic_youtube_search_retriever_chain,
    create_basic_youtube_search_answering_chain,
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
def sample_youtube_results():
    return {
        "results": [
            {
                "title": "Python Tutorial for Beginners",
                "url": "https://youtube.com/watch?v=1",
                "content": "Learn Python programming basics",
                "img_src": "https://i.ytimg.com/vi/1/default.jpg"
            },
            {
                "title": "Advanced Python Concepts",
                "url": "https://youtube.com/watch?v=2",
                "content": "Advanced Python programming techniques",
                "img_src": "https://i.ytimg.com/vi/2/default.jpg"
            }
        ]
    }

@pytest.mark.asyncio
async def test_handle_youtube_search_basic(mock_chat_model, mock_embeddings_model):
    query = "Python programming tutorials"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!")
    ]

    with patch('backend.agents.youtube_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {
            "results": [
                {
                    "title": "Python Tutorial",
                    "url": "https://youtube.com/watch?v=1",
                    "content": "Learn Python basics",
                    "img_src": "https://i.ytimg.com/vi/1/default.jpg"
                }
            ]
        }
        mock_chat_model.arun.return_value = "Here's a great Python tutorial video [1]"

        async for result in handle_youtube_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "response":
                assert "[1]" in result["data"]
                assert "Python" in result["data"]
            elif result["type"] == "sources":
                assert len(result["data"]) > 0
                assert "url" in result["data"][0]
                assert "img_src" in result["data"][0]

@pytest.mark.asyncio
async def test_retriever_chain_creation(mock_chat_model):
    chain = await create_basic_youtube_search_retriever_chain(mock_chat_model)
    assert isinstance(chain, RunnableSequence)
    
    with patch('backend.agents.youtube_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        mock_chat_model.arun.return_value = "rephrased query"
        
        result = await chain.invoke({
            "query": "test query",
            "chat_history": []
        })
        
        assert "query" in result
        assert "docs" in result

@pytest.mark.asyncio
async def test_optimization_modes(mock_chat_model, mock_embeddings_model, sample_youtube_results):
    query = "test query"
    history = []
    
    with patch('backend.agents.youtube_search_agent.search_searxng') as mock_search:
        mock_search.return_value = sample_youtube_results
        
        for mode in ["speed", "balanced", "quality"]:
            async for result in handle_youtube_search(
                query, history, mock_chat_model, mock_embeddings_model, mode
            ):
                if result["type"] == "sources":
                    assert isinstance(result["data"], list)
                elif result["type"] == "response":
                    assert isinstance(result["data"], str)

@pytest.mark.asyncio
async def test_query_rephrasing(mock_chat_model, mock_embeddings_model):
    query = "Show me how to make a cake"
    history = [
        HumanMessage(content="I want to learn cooking"),
        AIMessage(content="I can help with that!")
    ]

    with patch('backend.agents.youtube_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        mock_chat_model.arun.return_value = "cake making tutorial"

        async for _ in handle_youtube_search(query, history, mock_chat_model, mock_embeddings_model):
            pass
        
        # Verify the query was processed through the rephrasing chain
        assert mock_chat_model.arun.called
        call_args = mock_chat_model.arun.call_args[1]
        assert "chat_history" in call_args
        assert "query" in call_args

@pytest.mark.asyncio
async def test_youtube_specific_search(mock_chat_model, mock_embeddings_model):
    query = "test query"
    history = []

    with patch('backend.agents.youtube_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        
        async for _ in handle_youtube_search(query, history, mock_chat_model, mock_embeddings_model):
            pass
        
        # Verify YouTube-specific search configuration
        mock_search.assert_called_once()
        call_args = mock_search.call_args[1]
        assert call_args.get("engines") == ["youtube"]

@pytest.mark.asyncio
async def test_error_handling(mock_chat_model, mock_embeddings_model):
    query = "test query"
    history = []

    with patch('backend.agents.youtube_search_agent.search_searxng') as mock_search:
        mock_search.side_effect = Exception("Search API error")
        
        async for result in handle_youtube_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "error":
                assert "error" in result["data"].lower()

@pytest.mark.asyncio
async def test_empty_query_handling(mock_chat_model, mock_embeddings_model):
    query = ""
    history = []

    async for result in handle_youtube_search(query, history, mock_chat_model, mock_embeddings_model):
        if result["type"] == "response":
            assert "specific question" in result["data"].lower()

@pytest.mark.asyncio
async def test_source_tracking(mock_chat_model, mock_embeddings_model, sample_youtube_results):
    query = "test query"
    history = []

    with patch('backend.agents.youtube_search_agent.search_searxng') as mock_search:
        mock_search.return_value = sample_youtube_results
        mock_chat_model.arun.return_value = "Response with citation [1][2]"

        sources_received = False
        response_received = False
        
        async for result in handle_youtube_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "sources":
                sources_received = True
                assert len(result["data"]) == len(sample_youtube_results["results"])
                for source in result["data"]:
                    assert "title" in source
                    assert "url" in source
                    assert "img_src" in source
            elif result["type"] == "response":
                response_received = True
                assert "[1]" in result["data"]
                assert "[2]" in result["data"]
        
        assert sources_received and response_received

@pytest.mark.asyncio
async def test_invalid_history_handling(mock_chat_model, mock_embeddings_model):
    query = "test query"
    invalid_history = ["not a valid message"]  # Invalid history format

    with patch('backend.agents.youtube_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        
        async for result in handle_youtube_search(query, invalid_history, mock_chat_model, mock_embeddings_model):
            # Should handle invalid history gracefully
            assert result["type"] in ["response", "sources", "error"]

@pytest.mark.asyncio
async def test_malformed_search_results(mock_chat_model, mock_embeddings_model):
    query = "test query"
    history = []

    with patch('backend.agents.youtube_search_agent.search_searxng') as mock_search:
        # Missing required fields in results
        mock_search.return_value = {
            "results": [
                {"title": "Test"},  # Missing url and content
                {"url": "https://youtube.com"},  # Missing title and content
                {
                    "title": "Complete",
                    "url": "https://youtube.com/complete",
                    "content": "Complete content",
                    "img_src": "https://i.ytimg.com/vi/3/default.jpg"
                }
            ]
        }
        
        async for result in handle_youtube_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "sources":
                # Should handle malformed results gracefully
                assert len(result["data"]) > 0
