import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain.schema import HumanMessage, AIMessage, Document
from backend.agents.academic_search_agent import (
    handle_academic_search,
    create_basic_academic_search_retriever_chain,
    rerank_docs,
    process_docs,
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
def sample_documents():
    return [
        Document(
            page_content="Quantum computing research paper",
            metadata={"title": "Quantum Paper", "url": "http://example.com/1"}
        ),
        Document(
            page_content="Machine learning study",
            metadata={"title": "ML Study", "url": "http://example.com/2"}
        )
    ]

@pytest.mark.asyncio
async def test_handle_academic_search_basic(mock_chat_model, mock_embeddings_model):
    query = "Recent advancements in quantum computing"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I help you with academic research today?")
    ]

    with patch('backend.agents.academic_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {
            "results": [
                {"title": "Quantum Paper", "url": "http://example.com", "content": "Latest developments"}
            ]
        }
        mock_chat_model.arun.return_value = "Recent advancements include qubit improvements"

        async for result in handle_academic_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "response":
                assert "qubit" in result["data"].lower()
            elif result["type"] == "sources":
                assert len(result["data"]) > 0
                assert "title" in result["data"][0]

@pytest.mark.asyncio
async def test_handle_academic_search_empty_query(mock_chat_model, mock_embeddings_model):
    query = ""
    history = []

    async for result in handle_academic_search(query, history, mock_chat_model, mock_embeddings_model):
        if result["type"] == "response":
            assert "not sure how to help" in result["data"].lower()

@pytest.mark.asyncio
async def test_handle_academic_search_search_error(mock_chat_model, mock_embeddings_model):
    query = "quantum computing"
    history = []

    with patch('backend.agents.academic_search_agent.search_searxng') as mock_search:
        mock_search.side_effect = Exception("Search API error")

        async for result in handle_academic_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "error":
                assert "error" in result["data"].lower()

@pytest.mark.asyncio
async def test_rerank_docs_speed_mode(mock_embeddings_model):
    query = "quantum computing"
    docs = [Document(page_content=f"Content {i}") for i in range(20)]
    
    result = await rerank_docs(query, docs, mock_embeddings_model, "speed")
    assert len(result) == 15  # Speed mode should return top 15 docs without reranking

@pytest.mark.asyncio
async def test_rerank_docs_balanced_mode(mock_embeddings_model):
    query = "quantum computing"
    docs = [Document(page_content=f"Content {i}") for i in range(5)]
    
    result = await rerank_docs(query, docs, mock_embeddings_model, "balanced")
    assert len(result) == 5
    mock_embeddings_model.embed_documents.assert_called_once()
    mock_embeddings_model.embed_query.assert_called_once()

@pytest.mark.asyncio
async def test_process_docs():
    docs = [
        Document(page_content="First document"),
        Document(page_content="Second document")
    ]
    result = await process_docs(docs)
    assert "1. First document" in result
    assert "2. Second document" in result

@pytest.mark.asyncio
async def test_create_retriever_chain(mock_chat_model):
    chain = await create_basic_academic_search_retriever_chain(mock_chat_model)
    assert isinstance(chain, RunnableSequence)
    
    mock_chat_model.arun.return_value = "rephrased query"
    result = await chain.invoke({"query": "test query", "chat_history": []})
    assert "query" in result
    assert "docs" in result

@pytest.mark.asyncio
async def test_handle_academic_search_with_optimization_modes(mock_chat_model, mock_embeddings_model):
    query = "quantum computing"
    history = []
    
    for mode in ["speed", "balanced", "quality"]:
        with patch('backend.agents.academic_search_agent.search_searxng') as mock_search:
            mock_search.return_value = {
                "results": [{"title": "Test", "url": "http://test.com", "content": "Test content"}]
            }
            
            async for result in handle_academic_search(
                query, history, mock_chat_model, mock_embeddings_model, mode
            ):
                if result["type"] == "response":
                    assert isinstance(result["data"], str)
                elif result["type"] == "sources":
                    assert isinstance(result["data"], list)

@pytest.mark.asyncio
async def test_integration_search_and_response(mock_chat_model, mock_embeddings_model):
    """Test the integration between search and response generation"""
    query = "quantum computing advances"
    history = []
    
    with patch('backend.agents.academic_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {
            "results": [
                {
                    "title": "Recent Quantum Computing Paper",
                    "url": "http://example.com",
                    "content": "Breakthrough in quantum computing achieved"
                }
            ]
        }
        
        sources_received = False
        response_received = False
        
        async for result in handle_academic_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "sources":
                sources_received = True
                assert len(result["data"]) > 0
            elif result["type"] == "response":
                response_received = True
                assert isinstance(result["data"], str)
        
        assert sources_received and response_received

@pytest.mark.asyncio
async def test_error_handling_invalid_optimization_mode(mock_chat_model, mock_embeddings_model):
    query = "test query"
    history = []
    
    with pytest.raises(Exception):
        async for _ in handle_academic_search(
            query, history, mock_chat_model, mock_embeddings_model, "invalid_mode"
        ):
            pass

@pytest.mark.asyncio
async def test_empty_search_results(mock_chat_model, mock_embeddings_model):
    query = "very specific query with no results"
    history = []
    
    with patch('backend.agents.academic_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        
        async for result in handle_academic_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "response":
                assert "could not find" in result["data"].lower()
            elif result["type"] == "sources":
                assert len(result["data"]) == 0
