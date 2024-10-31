import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain.schema import HumanMessage, AIMessage, Document
from backend.agents.web_search_agent import (
    handle_web_search,
    create_basic_web_search_retriever_chain,
    create_basic_web_search_answering_chain,
    RunnableSequence
)

@pytest.fixture
def mock_chat_model():
    mock = AsyncMock()
    mock.arun = AsyncMock()
    mock.agenerate = AsyncMock()
    return mock

@pytest.fixture
def mock_embeddings_model():
    mock = AsyncMock()
    mock.embed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock.embed_query = AsyncMock(return_value=[0.2, 0.3, 0.4])
    return mock

@pytest.fixture
def sample_web_results():
    return {
        "results": [
            {
                "title": "Paris Info",
                "url": "https://example.com/paris",
                "content": "Paris is the capital of France"
            },
            {
                "title": "France Guide",
                "url": "https://example.com/france",
                "content": "Paris has a population of 2.2 million"
            }
        ]
    }

@pytest.mark.asyncio
async def test_handle_web_search_basic(mock_chat_model, mock_embeddings_model):
    query = "What is the capital of France?"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!")
    ]

    with patch('backend.agents.web_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {
            "results": [
                {
                    "title": "Paris Info",
                    "url": "https://example.com/paris",
                    "content": "Paris is the capital of France"
                }
            ]
        }
        mock_chat_model.arun.return_value = "The capital of France is Paris [1]"

        async for result in handle_web_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "response":
                assert "[1]" in result["data"]
                assert "Paris" in result["data"]
            elif result["type"] == "sources":
                assert len(result["data"]) > 0
                assert "url" in result["data"][0]

@pytest.mark.asyncio
async def test_query_rephrasing_xml_format(mock_chat_model, mock_embeddings_model):
    query = "Can you summarize https://example.com/article"
    history = []

    with patch('backend.agents.web_search_agent.search_searxng') as mock_search, \
         patch('backend.lib.link_document.get_documents_from_links') as mock_get_docs:
        
        mock_chat_model.arun.side_effect = [
            """
            <question>
            summarize
            </question>

            <links>
            https://example.com/article
            </links>
            """,
            "Summary of the article"
        ]
        
        mock_get_docs.return_value = [
            Document(
                page_content="Article content",
                metadata={"title": "Article", "url": "https://example.com/article"}
            )
        ]
        
        mock_chat_model.agenerate.return_value.generations = [[MagicMock(text="Summarized content")]]

        async for result in handle_web_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "response":
                assert isinstance(result["data"], str)

@pytest.mark.asyncio
async def test_webpage_summarization(mock_chat_model, mock_embeddings_model):
    query = "Summarize https://example.com/article"
    history = []

    with patch('backend.lib.link_document.get_documents_from_links') as mock_get_docs:
        mock_get_docs.return_value = [
            Document(
                page_content="Long article content here",
                metadata={"title": "Test Article", "url": "https://example.com/article"}
            )
        ]
        
        mock_chat_model.arun.side_effect = [
            "<question>summarize</question>\n<links>https://example.com/article</links>",
            "Article summary"
        ]
        mock_chat_model.agenerate.return_value.generations = [[MagicMock(text="Summarized content")]]

        async for result in handle_web_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "response":
                assert isinstance(result["data"], str)
            elif result["type"] == "sources":
                assert len(result["data"]) > 0
                assert result["data"][0]["url"] == "https://example.com/article"

@pytest.mark.asyncio
async def test_optimization_modes(mock_chat_model, mock_embeddings_model, sample_web_results):
    query = "test query"
    history = []
    
    with patch('backend.agents.web_search_agent.search_searxng') as mock_search:
        mock_search.return_value = sample_web_results
        
        for mode in ["speed", "balanced", "quality"]:
            async for result in handle_web_search(
                query, history, mock_chat_model, mock_embeddings_model, mode
            ):
                if result["type"] == "sources":
                    assert isinstance(result["data"], list)
                elif result["type"] == "response":
                    assert isinstance(result["data"], str)

@pytest.mark.asyncio
async def test_error_handling(mock_chat_model, mock_embeddings_model):
    query = "test query"
    history = []

    with patch('backend.agents.web_search_agent.search_searxng') as mock_search:
        mock_search.side_effect = Exception("Search API error")
        
        async for result in handle_web_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "error":
                assert "error" in result["data"].lower()

@pytest.mark.asyncio
async def test_empty_query_handling(mock_chat_model, mock_embeddings_model):
    query = ""
    history = []

    async for result in handle_web_search(query, history, mock_chat_model, mock_embeddings_model):
        if result["type"] == "response":
            assert "specific question" in result["data"].lower()

@pytest.mark.asyncio
async def test_source_tracking(mock_chat_model, mock_embeddings_model, sample_web_results):
    query = "test query"
    history = []

    with patch('backend.agents.web_search_agent.search_searxng') as mock_search:
        mock_search.return_value = sample_web_results
        mock_chat_model.arun.return_value = "Response with citation [1][2]"

        sources_received = False
        response_received = False
        
        async for result in handle_web_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "sources":
                sources_received = True
                assert len(result["data"]) == len(sample_web_results["results"])
                for source in result["data"]:
                    assert "title" in source
                    assert "url" in source
            elif result["type"] == "response":
                response_received = True
                assert "[1]" in result["data"]
                assert "[2]" in result["data"]
        
        assert sources_received and response_received

@pytest.mark.asyncio
async def test_link_processing_error(mock_chat_model, mock_embeddings_model):
    query = "Summarize https://example.com/broken-link"
    history = []

    with patch('backend.lib.link_document.get_documents_from_links') as mock_get_docs:
        mock_get_docs.side_effect = Exception("Failed to fetch document")
        mock_chat_model.arun.return_value = "<question>summarize</question>\n<links>https://example.com/broken-link</links>"

        async for result in handle_web_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "error":
                assert "error" in result["data"].lower()

@pytest.mark.asyncio
async def test_multiple_link_summarization(mock_chat_model, mock_embeddings_model):
    query = "Summarize https://example.com/1 and https://example.com/2"
    history = []

    with patch('backend.lib.link_document.get_documents_from_links') as mock_get_docs:
        mock_get_docs.return_value = [
            Document(
                page_content="Content 1",
                metadata={"title": "Article 1", "url": "https://example.com/1"}
            ),
            Document(
                page_content="Content 2",
                metadata={"title": "Article 2", "url": "https://example.com/2"}
            )
        ]
        
        mock_chat_model.arun.side_effect = [
            "<question>summarize</question>\n<links>\nhttps://example.com/1\nhttps://example.com/2\n</links>",
            "Combined summary"
        ]
        mock_chat_model.agenerate.return_value.generations = [[MagicMock(text="Summarized content")]]

        sources_seen = set()
        async for result in handle_web_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "sources":
                for source in result["data"]:
                    sources_seen.add(source["url"])
        
        assert "https://example.com/1" in sources_seen
        assert "https://example.com/2" in sources_seen

@pytest.mark.asyncio
async def test_malformed_search_results(mock_chat_model, mock_embeddings_model):
    query = "test query"
    history = []

    with patch('backend.agents.web_search_agent.search_searxng') as mock_search:
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
        
        async for result in handle_web_search(query, history, mock_chat_model, mock_embeddings_model):
            if result["type"] == "sources":
                # Should handle malformed results gracefully
                assert len(result["data"]) > 0
