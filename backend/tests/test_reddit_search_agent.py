"""Unit tests for the Reddit Search Agent module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from agents.reddit_search_agent import RedditSearchAgent, handle_reddit_search
from utils.logger import logger

class TestRedditSearchAgent:
    """Test suite for RedditSearchAgent"""

    @pytest.fixture
    def mock_llm(self):
        """Fixture providing a mock LLM instance"""
        llm = MagicMock(spec=BaseChatModel)
        llm.ainvoke = AsyncMock(return_value="test query")
        return llm

    @pytest.fixture
    def mock_embeddings(self):
        """Fixture providing a mock embeddings instance"""
        return MagicMock(spec=Embeddings)

    @pytest.fixture
    def reddit_search_agent(self, mock_llm, mock_embeddings):
        """Fixture providing a RedditSearchAgent instance"""
        return RedditSearchAgent(mock_llm, mock_embeddings, "balanced")

    @pytest.fixture
    def mock_search_results(self):
        """Fixture providing standard search results"""
        return {
            "results": [{
                "content": "test content",
                "title": "test title", 
                "url": "http://test.com",
                "img_src": "http://test.com/image.jpg"
            }]
        }

    @pytest.fixture
    def mock_empty_search_results(self):
        """Fixture providing empty search results"""
        return {"results": []}

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_valid_query(self, reddit_search_agent, mock_search_results):
        """Test retriever chain creation with valid query"""
        with patch("agents.reddit_search_agent.search_searxng",
                  AsyncMock(return_value=mock_search_results)):
            chain = await reddit_search_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["query"] == "test query"
            assert len(result["docs"]) == 1
            assert result["docs"][0].page_content == "test content"
            assert result["docs"][0].metadata["title"] == "test title"

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_not_needed_query(self, reddit_search_agent):
        """Test retriever chain with query marked as not needed"""
        reddit_search_agent.llm.ainvoke = AsyncMock(return_value="not_needed")
        
        chain = await reddit_search_agent.create_retriever_chain()
        result = await chain.ainvoke({
            "query": "hi",
            "chat_history": []
        })
        
        assert result["query"] == ""
        assert len(result["docs"]) == 0

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_search_error(self, reddit_search_agent):
        """Test error handling in retriever chain"""
        with patch("agents.reddit_search_agent.search_searxng",
                  AsyncMock(side_effect=Exception("test error"))), \
             patch.object(logger, "error") as mock_logger:
            
            chain = await reddit_search_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["query"] == "test query"
            assert len(result["docs"]) == 0
            mock_logger.assert_called_once_with("Error in reddit search: %s", "test error")

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_empty_results(self, reddit_search_agent, mock_empty_search_results):
        """Test retriever chain with empty search results"""
        with patch("agents.reddit_search_agent.search_searxng",
                  AsyncMock(return_value=mock_empty_search_results)):
            chain = await reddit_search_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["query"] == "test query"
            assert len(result["docs"]) == 0

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_different_search_modes(self, mock_llm, mock_embeddings):
        """Test retriever chain with different search modes"""
        for mode in ["balanced", "precise", "fast"]:
            agent = RedditSearchAgent(mock_llm, mock_embeddings, mode)
            chain = await agent.create_retriever_chain()
            assert chain is not None

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_valid_response(self, reddit_search_agent):
        """Test answering chain with valid response"""
        mock_retriever_result = {
            "query": "test query",
            "docs": [MagicMock(
                page_content="test content",
                metadata={"title": "test title", "url": "http://test.com"}
            )]
        }
        
        mock_response = MagicMock()
        mock_response.content = "test response"
        reddit_search_agent.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch.object(reddit_search_agent, "create_retriever_chain",
                         AsyncMock(return_value=AsyncMock(ainvoke=AsyncMock(
                             return_value=mock_retriever_result
                         )))):
            chain = await reddit_search_agent.create_answering_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["type"] == "response"
            assert result["data"] == "test response"
            assert len(result["sources"]) == 1
            assert result["sources"][0]["title"] == "test title"

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_empty_query(self, reddit_search_agent):
        """Test answering chain with empty query"""
        mock_retriever_result = {
            "query": "",
            "docs": []
        }
        
        with patch.object(reddit_search_agent, "create_retriever_chain",
                         AsyncMock(return_value=AsyncMock(ainvoke=AsyncMock(
                             return_value=mock_retriever_result
                         )))):
            chain = await reddit_search_agent.create_answering_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["type"] == "response"
            assert "not sure how to help" in result["data"].lower()
            assert len(result["sources"]) == 0

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_error(self, reddit_search_agent):
        """Test error handling in answering chain"""
        with patch.object(reddit_search_agent, "create_retriever_chain",
                         AsyncMock(return_value=AsyncMock(ainvoke=AsyncMock(
                             side_effect=Exception("test error")
                         )))), \
             patch.object(logger, "error") as mock_logger:
            
            chain = await reddit_search_agent.create_answering_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["type"] == "response"
            assert "error occurred" in result["data"].lower()
            assert len(result["sources"]) == 0
            mock_logger.assert_called_with("Error in response generation: %s", "test error")

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_rate_limit_error(self, reddit_search_agent):
        """Test rate limit handling in answering chain"""
        with patch.object(reddit_search_agent, "create_retriever_chain",
                         AsyncMock(return_value=AsyncMock(ainvoke=AsyncMock(
                             side_effect=Exception("Rate limit exceeded")
                         )))):
            
            chain = await reddit_search_agent.create_answering_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["type"] == "response"
            assert "rate limit" in result["data"].lower()
            # Rate limit responses don't log errors since they're expected

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_invalid_input(self, reddit_search_agent):
        """Test answering chain with invalid input types"""
        chain = await reddit_search_agent.create_answering_chain()
        with pytest.raises(TypeError):
            await chain.ainvoke({
                "query": None,
                "chat_history": "invalid"
            })

    @pytest.mark.asyncio
    async def test_handle_reddit_search(self, mock_llm, mock_embeddings):
        """Test the handle_reddit_search function"""
        mock_agent = MagicMock()
        
        async def mock_gen(query, history):
            yield {
                "type": "response",
                "data": "test response"
            }
            
        mock_agent.handle_search = mock_gen
        
        with patch("agents.reddit_search_agent.RedditSearchAgent", return_value=mock_agent):
            gen = handle_reddit_search(
                query="test",
                history=[],
                llm=mock_llm,
                embeddings=mock_embeddings
            )
            
            results = [result async for result in gen]
            assert len(results) == 1
            assert results[0]["type"] == "response"
            assert results[0]["data"] == "test response"

    @pytest.mark.asyncio
    async def test_format_prompt(self, reddit_search_agent):
        """Test the prompt formatting"""
        mock_prompt = "base prompt"
        reddit_search_agent.create_base_prompt = MagicMock(return_value=mock_prompt)
        prompt = reddit_search_agent.format_prompt("test query", "test context")
        
        assert mock_prompt in prompt
        assert "Reddit" in prompt
        assert "test context" in prompt