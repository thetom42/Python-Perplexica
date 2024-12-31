import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from agents.academic_search_agent import AcademicSearchAgent
from utils.logger import logger

class TestAcademicSearchAgent:
    """Test suite for AcademicSearchAgent"""

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
    def academic_search_agent(self, mock_llm, mock_embeddings):
        """Fixture providing an AcademicSearchAgent instance"""
        return AcademicSearchAgent(mock_llm, mock_embeddings, "balanced")

    @pytest.fixture
    def mock_search_results(self):
        """Fixture providing standard search results"""
        return {
            "results": [{
                "content": "test content",
                "title": "test title",
                "url": "http://test.com",
                "img_src": "http://test.com/image.png"
            }]
        }

    @pytest.fixture
    def mock_empty_search_results(self):
        """Fixture providing empty search results"""
        return {"results": []}

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_valid_query(self, academic_search_agent, mock_search_results):
        """Test retriever chain creation with valid query"""
        with patch("agents.academic_search_agent.search_searxng", 
                  AsyncMock(return_value=mock_search_results)):
            chain = await academic_search_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["query"] == "test query"
            assert len(result["docs"]) == 1
            assert result["docs"][0].page_content == "test content"
            assert result["docs"][0].metadata["title"] == "test title"

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_not_needed_query(self, academic_search_agent):
        """Test retriever chain with query marked as not needed"""
        academic_search_agent.llm.ainvoke = AsyncMock(return_value="not_needed")
        
        chain = await academic_search_agent.create_retriever_chain()
        result = await chain.ainvoke({
            "query": "hi",
            "chat_history": []
        })
        
        assert result["query"] == ""
        assert len(result["docs"]) == 0

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_search_error(self, academic_search_agent):
        """Test error handling in retriever chain"""
        with patch("agents.academic_search_agent.search_searxng", 
                  AsyncMock(side_effect=Exception("test error"))), \
             patch.object(logger, "error") as mock_logger:
            
            chain = await academic_search_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["query"] == "test query"
            assert len(result["docs"]) == 0
            mock_logger.assert_called_once_with("Error in academic search: %s", "test error")

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_empty_results(self, academic_search_agent, mock_empty_search_results):
        """Test retriever chain with empty search results"""
        with patch("agents.academic_search_agent.search_searxng", 
                  AsyncMock(return_value=mock_empty_search_results)):
            chain = await academic_search_agent.create_retriever_chain()
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
            agent = AcademicSearchAgent(mock_llm, mock_embeddings, mode)
            chain = await agent.create_retriever_chain()
            assert chain is not None

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_valid_response(self, academic_search_agent):
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
        academic_search_agent.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch.object(academic_search_agent, "create_retriever_chain",
                         AsyncMock(return_value=AsyncMock(ainvoke=AsyncMock(
                             return_value=mock_retriever_result
                         )))):
            chain = await academic_search_agent.create_answering_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["response"] == "test response"
            assert len(result["sources"]) == 1
            assert result["sources"][0]["title"] == "test title"

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_empty_query(self, academic_search_agent):
        """Test answering chain with empty query"""
        mock_retriever_result = {
            "query": "",
            "docs": []
        }
        
        with patch.object(academic_search_agent, "create_retriever_chain",
                         AsyncMock(return_value=AsyncMock(ainvoke=AsyncMock(
                             return_value=mock_retriever_result
                         )))):
            chain = await academic_search_agent.create_answering_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["response"] == "I'm not sure how to help with that. Could you please ask a specific question?"
            assert len(result["sources"]) == 0

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_error(self, academic_search_agent):
        """Test error handling in answering chain"""
        with patch.object(academic_search_agent, "create_retriever_chain",
                         AsyncMock(return_value=AsyncMock(ainvoke=AsyncMock(
                             side_effect=Exception("test error")
                         )))), \
             patch.object(logger, "error") as mock_logger:
            
            chain = await academic_search_agent.create_answering_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["response"] == "An error occurred while processing the academic search results"
            assert len(result["sources"]) == 0
            mock_logger.assert_called_with("Error in response generation: %s", "test error")

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_rate_limit_error(self, academic_search_agent):
        """Test rate limit handling in answering chain"""
        with patch.object(academic_search_agent, "create_retriever_chain",
                         AsyncMock(return_value=AsyncMock(ainvoke=AsyncMock(
                             side_effect=Exception("Rate limit exceeded")
                         )))), \
             patch.object(logger, "error") as mock_logger:
            
            chain = await academic_search_agent.create_answering_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert "rate limit" in result["response"].lower()
            mock_logger.assert_called_with("Error in response generation: %s", "Rate limit exceeded")

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_invalid_input(self, academic_search_agent):
        """Test answering chain with invalid input types"""
        with patch.object(academic_search_agent, "create_retriever_chain",
                       AsyncMock(return_value=AsyncMock(ainvoke=AsyncMock(
                           side_effect=TypeError("Invalid input types")
                       )))):
            chain = await academic_search_agent.create_answering_chain()
            result = await chain.ainvoke({
                "query": None,
                "chat_history": "invalid"
            })
            
            assert result["response"] == "An error occurred while processing the academic search results"
            assert len(result["sources"]) == 0