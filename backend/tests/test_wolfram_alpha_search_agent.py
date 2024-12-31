"""
Unit tests for the Wolfram Alpha Search Agent module.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain.schema import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableSequence
from langchain_core.outputs import Generation, ChatGeneration
from langchain_core.messages import AIMessage
from agents.wolfram_alpha_search_agent import WolframAlphaSearchAgent, handle_wolfram_alpha_search
from utils.logger import logger

class TestWolframAlphaSearchAgent:
    """Test suite for WolframAlphaSearchAgent"""
    @pytest.fixture
    def mock_llm(self):
        """Fixture providing a mock LLM instance"""
        llm = MagicMock(spec=BaseChatModel)
        
        # Configure mock responses to return valid Generation objects
        async def mock_invoke(x, **kwargs):
            text = x["query"] if isinstance(x, dict) else x
            return ChatGeneration(message=AIMessage(content=str(text)))
        
        async def mock_ainvoke(x, **kwargs):
            text = x["query"] if isinstance(x, dict) else x
            return ChatGeneration(message=AIMessage(content=str(text)))
        
        llm.invoke = AsyncMock(side_effect=mock_invoke)
        llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        
        # Support for | operator in chain composition
        llm.__or__ = lambda self, other: RunnableSequence.from_([self, other])
        return llm

    @pytest.fixture
    def mock_embeddings(self):
        """Fixture providing a mock embeddings instance"""
        return MagicMock(spec=Embeddings)

    @pytest.fixture
    def wolfram_alpha_search_agent(self, mock_llm, mock_embeddings):
        """Fixture providing a WolframAlphaSearchAgent instance"""
        return WolframAlphaSearchAgent(mock_llm, mock_embeddings)

    @pytest.fixture
    def mock_search_results(self):
        """Fixture providing standard search results"""
        return {
            "results": [{
                "content": "Test content",
                "title": "Test title",
                "url": "https://example.com"
            }]
        }

    @pytest.fixture
    def mock_empty_search_results(self):
        """Fixture providing empty search results"""
        return {"results": []}

    @pytest.mark.asyncio
    async def test_create_retriever_chain(self, wolfram_alpha_search_agent):
        """Test creation of retriever chain"""
        mock_chain = AsyncMock()
        mock_chain.invoke = AsyncMock()
        
        with patch('agents.wolfram_alpha_search_agent.RunnableSequence', return_value=mock_chain):
            chain = await wolfram_alpha_search_agent.create_retriever_chain()
            assert chain is not None

    @pytest.mark.asyncio
    async def test_process_output_with_valid_query(self, wolfram_alpha_search_agent, mock_search_results):
        """Test process_output with a valid Wolfram Alpha query"""
        # Mock LLM response to return a valid Generation object
        wolfram_alpha_search_agent.llm.ainvoke = AsyncMock(return_value=Generation(text="Atomic radius of S"))
        
        # Mock search results
        with patch('lib.searxng.search_searxng', AsyncMock(return_value=mock_search_results)):
            chain = await wolfram_alpha_search_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "What is the atomic radius of S?",
                "chat_history": []
            })
            
            assert "query" in result
            assert "docs" in result
            assert len(result["docs"]) == 1
            assert result["docs"][0].page_content == "Test content"

    @pytest.mark.asyncio
    async def test_process_output_with_not_needed(self, wolfram_alpha_search_agent):
        """Test process_output when the query is not needed"""
        # Mock LLM to return a valid Generation object
        wolfram_alpha_search_agent.llm.invoke = AsyncMock(return_value=Generation(text="not_needed"))
        wolfram_alpha_search_agent.llm.ainvoke = AsyncMock(return_value=Generation(text="not_needed"))
        
        chain = await wolfram_alpha_search_agent.create_retriever_chain()
        result = await chain.ainvoke({
            "query": "Hello",
            "chat_history": []
        })
        
        assert result["query"] == ""
        assert result["docs"] == []

    @pytest.mark.asyncio
    async def test_create_answering_chain(self, wolfram_alpha_search_agent):
        """Test creation of answering chain"""
        mock_chain = AsyncMock()
        mock_chain.invoke = AsyncMock()
        
        with patch('agents.wolfram_alpha_search_agent.RunnableSequence', return_value=mock_chain):
            chain = await wolfram_alpha_search_agent.create_answering_chain()
            assert chain is not None

    @pytest.mark.asyncio
    async def test_handle_wolfram_alpha_search(self):
        """Test the handle_wolfram_alpha_search function"""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_embeddings = MagicMock(spec=Embeddings)
        
        async for result in handle_wolfram_alpha_search(
            query="test",
            history=[],
            llm=mock_llm,
            embeddings=mock_embeddings
        ):
            assert result is not None

    @pytest.mark.asyncio
    async def test_error_handling(self, wolfram_alpha_search_agent):
        """Test error handling in the Wolfram Alpha search process"""
        # Mock LLM to raise an exception
        wolfram_alpha_search_agent.llm.invoke.side_effect = Exception("Test error")
        
        chain = await wolfram_alpha_search_agent.create_retriever_chain()
        result = await chain.ainvoke({
            "query": "test",
            "chat_history": []
        })
        
        # Should return empty result on error
        assert result["query"] == "test"
        assert result["docs"] == []

    @pytest.mark.asyncio
    async def test_format_prompt(self, wolfram_alpha_search_agent):
        """Test the format_prompt method"""
        query = "test query"
        context = "test context"
        prompt = wolfram_alpha_search_agent.format_prompt(query, context)
        assert isinstance(prompt, str)
        assert "Wolfram Alpha" in prompt
        assert "computational knowledge engine" in prompt
        assert "test context" in prompt
        assert "You are Perplexica" in prompt

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_search_error(self, wolfram_alpha_search_agent):
        """Test error handling in retriever chain"""
        with patch("lib.searxng.search_searxng", 
                  AsyncMock(side_effect=Exception("test error"))), \
             patch.object(logger, "error") as mock_logger:
            
            chain = await wolfram_alpha_search_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["query"] == ""
            assert len(result["docs"]) == 0
            mock_logger.assert_called_once_with("Error in Wolfram Alpha search: %s", "test error")

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_empty_results(self, wolfram_alpha_search_agent, mock_empty_search_results):
        """Test retriever chain with empty search results"""
        with patch("lib.searxng.search_searxng", 
                  AsyncMock(return_value=mock_empty_search_results)):
            chain = await wolfram_alpha_search_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["query"] == ""
            assert len(result["docs"]) == 0

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_valid_response(self, wolfram_alpha_search_agent):
        """Test answering chain with valid response"""
        # Mock retriever chain result
        mock_retriever_result = {
            "query": "test query",
            "docs": [Document(
                page_content="test content",
                metadata={"title": "test title", "url": "http://test.com"}
            )]
        }
        
        # Mock LLM response to return a valid Generation object
        wolfram_alpha_search_agent.llm.invoke = AsyncMock(return_value=Generation(text="test response"))
        wolfram_alpha_search_agent.llm.ainvoke = AsyncMock(return_value=Generation(text="test response"))
        
        # Mock retriever chain
        mock_retriever_chain = AsyncMock()
        mock_retriever_chain.invoke = AsyncMock(return_value=mock_retriever_result)
        
        with patch.object(wolfram_alpha_search_agent, "create_retriever_chain",
                         AsyncMock(return_value=mock_retriever_chain)):
            chain = await wolfram_alpha_search_agent.create_answering_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["type"] == "response"
            assert result["data"] == "test response"
            assert len(result["sources"]) == 1
            assert result["sources"][0]["title"] == "test title"

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_empty_query(self, wolfram_alpha_search_agent):
        """Test answering chain with empty query"""
        # Mock retriever chain result with empty query
        mock_retriever_result = {
            "query": "",
            "docs": [],
            "sources": []
        }
        
        # Mock retriever chain
        mock_retriever_chain = AsyncMock()
        mock_retriever_chain.invoke = AsyncMock(return_value=mock_retriever_result)
        
        with patch.object(wolfram_alpha_search_agent, "create_retriever_chain",
                         AsyncMock(return_value=mock_retriever_chain)):
            chain = await wolfram_alpha_search_agent.create_answering_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["type"] == "response"
            assert result["data"] == "I'm not sure how to help with that. Could you please ask a specific question?"

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_error(self, wolfram_alpha_search_agent):
        """Test error handling in answering chain"""
        # Mock retriever chain to raise an exception
        mock_retriever_chain = AsyncMock()
        mock_retriever_chain.invoke = AsyncMock(side_effect=Exception("test error"))
        
        with patch.object(wolfram_alpha_search_agent, "create_retriever_chain",
                         AsyncMock(return_value=mock_retriever_chain)), \
             patch.object(logger, "error") as mock_logger:
            
            chain = await wolfram_alpha_search_agent.create_answering_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["type"] == "error"
            assert result["data"] == "An error occurred while generating the response"
            mock_logger.assert_called_with("Error in response generation: %s", "test error")

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_rate_limit_error(self, wolfram_alpha_search_agent):
        """Test rate limit handling in answering chain"""
        # Mock retriever chain to raise a rate limit exception
        mock_retriever_chain = AsyncMock()
        mock_retriever_chain.invoke = AsyncMock(side_effect=Exception("Rate limit exceeded"))
        
        with patch.object(wolfram_alpha_search_agent, "create_retriever_chain",
                         AsyncMock(return_value=mock_retriever_chain)), \
             patch.object(logger, "error") as mock_logger:
            
            chain = await wolfram_alpha_search_agent.create_answering_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["type"] == "error"
            assert result["data"] == "An error occurred while generating the response"
            mock_logger.assert_called_with("Error in response generation: %s", "Rate limit exceeded")

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_invalid_input(self, wolfram_alpha_search_agent):
        """Test answering chain with invalid input types"""
        with pytest.raises(TypeError):
            chain = await wolfram_alpha_search_agent.create_answering_chain()
            await chain.ainvoke({
                "query": None,
                "chat_history": "invalid"
            })
