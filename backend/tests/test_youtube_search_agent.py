"""Unit tests for the YouTubeSearchAgent module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain.schema import BaseMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from agents.youtube_search_agent import YouTubeSearchAgent, BASIC_YOUTUBE_SEARCH_RETRIEVER_PROMPT
from utils.logger import logger

class TestYouTubeSearchAgent:
    """Test suite for YouTubeSearchAgent"""

    @pytest.fixture
    def mock_llm(self):
        """Fixture providing a mock LLM instance"""
        llm = MagicMock(spec=BaseChatModel)
        llm.ainvoke = AsyncMock()
        return llm

    @pytest.fixture
    def mock_embeddings(self):
        """Fixture providing a mock embeddings instance"""
        return MagicMock(spec=Embeddings)

    @pytest.fixture
    def youtube_agent(self, mock_llm, mock_embeddings):
        """Fixture providing a YouTubeSearchAgent instance"""
        agent = YouTubeSearchAgent(mock_llm, mock_embeddings, "balanced")
        # Mock the abstract methods from AbstractAgent
        agent.rerank_docs = AsyncMock()
        agent.process_docs = AsyncMock()
        agent.create_base_prompt = MagicMock(return_value="Base prompt")
        return agent

    @pytest.fixture
    def mock_search_results(self):
        """Fixture providing standard search results"""
        return {
            "results": [{
                "title": "Test Video",
                "content": "Test content",
                "url": "https://example.com",
                "img_src": "https://example.com/image.jpg"
            }]
        }

    @pytest.fixture
    def mock_empty_search_results(self):
        """Fixture providing empty search results"""
        return {"results": []}

    @pytest.mark.asyncio
    async def test_create_retriever_chain(self, youtube_agent):
        """Test the creation of the retriever chain"""
        chain = await youtube_agent.create_retriever_chain()
        assert chain is not None

    @pytest.mark.asyncio
    async def test_retriever_chain_with_valid_query(self, youtube_agent, mock_llm, mock_search_results):
        """Test retriever chain with valid query"""
        mock_llm.ainvoke.return_value = "test query"
        
        with patch("agents.youtube_search_agent.search_searxng", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_search_results
            
            chain = await youtube_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["query"] == "test query"
            assert len(result["docs"]) == 1
            assert isinstance(result["docs"][0], Document)

    @pytest.mark.asyncio
    async def test_retriever_chain_with_not_needed(self, youtube_agent, mock_llm):
        """Test retriever chain with 'not_needed' response"""
        mock_llm.ainvoke.return_value = "not_needed"
        
        chain = await youtube_agent.create_retriever_chain()
        result = await chain.ainvoke({
            "query": "hi",
            "chat_history": []
        })
        
        assert result["query"] == ""
        assert len(result["docs"]) == 0

    @pytest.mark.asyncio
    async def test_retriever_chain_with_search_error(self, youtube_agent, mock_llm):
        """Test error handling in retriever chain"""
        mock_llm.ainvoke.return_value = "test query"
        
        with patch("agents.youtube_search_agent.search_searxng", new_callable=AsyncMock) as mock_search, \
             patch.object(logger, "error") as mock_logger:
            mock_search.side_effect = Exception("Search failed")
            
            chain = await youtube_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["query"] == "test query"
            assert len(result["docs"]) == 0
            mock_logger.assert_called_once_with("Error in YouTube search: %s", "Search failed")

    @pytest.mark.asyncio
    async def test_retriever_chain_with_empty_results(self, youtube_agent, mock_llm, mock_empty_search_results):
        """Test retriever chain with empty search results"""
        mock_llm.ainvoke.return_value = "test query"
        
        with patch("agents.youtube_search_agent.search_searxng", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_empty_search_results
            
            chain = await youtube_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["query"] == "test query"
            assert len(result["docs"]) == 0

    @pytest.mark.asyncio
    async def test_create_answering_chain(self, youtube_agent):
        """Test the creation of the answering chain"""
        chain = await youtube_agent.create_answering_chain()
        assert chain is not None

    @pytest.mark.asyncio
    async def test_answering_chain_with_results(self, youtube_agent, mock_llm, mock_search_results):
        """Test answering chain with search results"""
        mock_llm.ainvoke.side_effect = [
            "test query",  # For retriever chain
            "Test response"  # For answering chain
        ]
        
        with patch("agents.youtube_search_agent.search_searxng", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_search_results
            
            youtube_agent.rerank_docs.return_value = [Document(page_content="Test content")]
            youtube_agent.process_docs.return_value = "Test context"
            
            chain = await youtube_agent.create_answering_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["type"] == "error"
            assert "An error occurred" in result["data"]
            assert len(result["sources"]) == 1

    @pytest.mark.asyncio
    async def test_answering_chain_with_document_processing_error(self, youtube_agent, mock_llm, mock_search_results):
        """Test error handling in answering chain when document processing fails"""
        mock_llm.ainvoke.return_value = "test query"
        
        with patch("agents.youtube_search_agent.search_searxng", new_callable=AsyncMock) as mock_search, \
             patch.object(logger, "error") as mock_logger:
            mock_search.return_value = mock_search_results
            
            youtube_agent.rerank_docs.side_effect = Exception("Processing failed")
            
            chain = await youtube_agent.create_answering_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["type"] == "error"
            assert "An error occurred" in result["data"]
            mock_logger.assert_called_once_with("Error in document processing: %s", "Processing failed")

    @pytest.mark.asyncio
    async def test_retriever_chain_with_rate_limit_error(self, youtube_agent, mock_llm):
        """Test rate limit handling in retriever chain"""
        mock_llm.ainvoke.return_value = "test query"
        
        with patch("agents.youtube_search_agent.search_searxng", new_callable=AsyncMock) as mock_search, \
             patch.object(logger, "error") as mock_logger:
            mock_search.side_effect = Exception("Rate limit exceeded")
            
            chain = await youtube_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["query"] == "test query"
            assert len(result["docs"]) == 0
            mock_logger.assert_called_once_with("Error in YouTube search: %s", "Rate limit exceeded")

    @pytest.mark.asyncio
    async def test_retriever_chain_with_different_search_modes(self, mock_llm, mock_embeddings):
        """Test retriever chain with different search modes"""
        for mode in ["balanced", "precise", "fast"]:
            agent = YouTubeSearchAgent(mock_llm, mock_embeddings, mode)
            chain = await agent.create_retriever_chain()
            assert chain is not None

    @pytest.mark.asyncio
    async def test_parse_response(self, youtube_agent):
        """Test response parsing"""
        response = "Test response"
        result = await youtube_agent.parse_response(response)
        
        assert result["type"] == "response"
        assert result["data"] == response

    @pytest.mark.asyncio
    async def test_parse_response_with_error(self, youtube_agent):
        """Test error handling in response parsing"""
        with patch.object(logger, "error") as mock_logger:
            result = await youtube_agent.parse_response(None)
            
            assert result["type"] == "error"
            assert "An error occurred" in result["data"]
            mock_logger.assert_called_once_with("Error parsing YouTube search response: %s", None)