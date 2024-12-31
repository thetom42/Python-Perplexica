"""
Unit tests for the Image Search Agent module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain.schema import HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda
from agents.image_search_agent import ImageSearchAgent, handle_image_search
from agents.abstract_agent import AbstractAgent
from utils.logger import logger

class TestImageSearchAgent:
    """Test suite for ImageSearchAgent"""

    @pytest.fixture
    def mock_llm(self):
        """Fixture providing a mock LLM instance"""
        llm = MagicMock(spec=BaseChatModel)
        llm.ainvoke = AsyncMock(return_value="rephrased query")
        return llm

    @pytest.fixture
    def mock_embeddings(self):
        """Fixture providing a mock embeddings instance"""
        return MagicMock(spec=Embeddings)

    @pytest.fixture
    def image_search_agent(self, mock_llm, mock_embeddings):
        """Fixture providing an ImageSearchAgent instance"""
        return ImageSearchAgent(mock_llm, mock_embeddings)

    @pytest.fixture
    def mock_search_results(self):
        """Fixture providing standard search results"""
        return {
            "results": [{
                "img_src": "http://example.com/image1.jpg",
                "url": "http://example.com",
                "title": "Example Image 1"
            }]
        }

    @pytest.fixture
    def mock_empty_search_results(self):
        """Fixture providing empty search results"""
        return {"results": []}

    @pytest.mark.asyncio
    async def test_create_retriever_chain(self, image_search_agent):
        """Test the creation of the retriever chain."""
        chain = await image_search_agent.create_retriever_chain()
        assert chain is not None

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_search_error(self, image_search_agent):
        """Test error handling in retriever chain"""
        with patch("agents.image_search_agent.search_searxng",
                  AsyncMock(side_effect=Exception("test error"))), \
             patch.object(logger, "error") as mock_logger:
            
            chain = await image_search_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["query"] == ""
            assert len(result["images"]) == 0
            mock_logger.assert_called_once_with("Error in image search: %s", "test error")

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_empty_results(self, image_search_agent, mock_empty_search_results):
        """Test retriever chain with empty search results"""
        with patch("agents.image_search_agent.search_searxng",
                  AsyncMock(return_value=mock_empty_search_results)):
            chain = await image_search_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["query"] == "rephrased query"
            assert len(result["images"]) == 0

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_invalid_query(self, image_search_agent):
        """Test retriever chain with invalid query"""
        with pytest.raises(TypeError):
            chain = await image_search_agent.create_retriever_chain()
            await chain.ainvoke({
                "query": None,
                "chat_history": "invalid"
            })

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_rate_limit_error(self, image_search_agent):
        """Test rate limit handling in retriever chain"""
        with patch("agents.image_search_agent.search_searxng",
                  AsyncMock(side_effect=Exception("Rate limit exceeded"))), \
             patch.object(logger, "error") as mock_logger:
            
            chain = await image_search_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert "rate limit" in result["response"].lower()
            mock_logger.assert_called_with("Error in image search: %s", "Rate limit exceeded")

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_different_configurations(self, mock_llm, mock_embeddings):
        """Test retriever chain with different configurations"""
        for config in ["default", "strict", "lenient"]:
            agent = ImageSearchAgent(mock_llm, mock_embeddings, config)
            chain = await agent.create_retriever_chain()
            assert chain is not None

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_large_input(self, image_search_agent):
        """Test retriever chain with large input"""
        large_query = "a" * 10000
        chain = await image_search_agent.create_retriever_chain()
        result = await chain.ainvoke({
            "query": large_query,
            "chat_history": []
        })
        assert len(result["query"]) < len(large_query)  # Should be processed/truncated

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_special_characters(self, image_search_agent):
        """Test retriever chain with special characters"""
        special_query = "!@#$%^&*()"
        chain = await image_search_agent.create_retriever_chain()
        result = await chain.ainvoke({
            "query": special_query,
            "chat_history": []
        })
        assert result["query"] != special_query  # Should be processed

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_empty_query(self, image_search_agent):
        """Test retriever chain with empty query"""
        chain = await image_search_agent.create_retriever_chain()
        result = await chain.ainvoke({
            "query": "",
            "chat_history": []
        })
        assert result["query"] == ""
        assert len(result["images"]) == 0

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_search_error(self, image_search_agent):
        """Test error handling in retriever chain"""
        with patch("agents.image_search_agent.search_searxng",
                  AsyncMock(side_effect=Exception("test error"))), \
             patch.object(logger, "error") as mock_logger:
            
            chain = await image_search_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["query"] == ""
            assert len(result["images"]) == 0
            mock_logger.assert_called_once_with("Error in image search: %s", "test error")

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_empty_results(self, image_search_agent, mock_empty_search_results):
        """Test retriever chain with empty search results"""
        with patch("agents.image_search_agent.search_searxng",
                  AsyncMock(return_value=mock_empty_search_results)):
            chain = await image_search_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert result["query"] == "rephrased query"
            assert len(result["images"]) == 0

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_invalid_query(self, image_search_agent):
        """Test retriever chain with invalid query"""
        with pytest.raises(TypeError):
            chain = await image_search_agent.create_retriever_chain()
            await chain.ainvoke({
                "query": None,
                "chat_history": "invalid"
            })

    @pytest.mark.asyncio
    async def test_process_output_success(self, image_search_agent):
        """Test successful processing of search results."""
        mock_chain = AsyncMock()
        mock_chain.invoke = AsyncMock(return_value={
            "query": "rephrased query",
            "images": [
                {
                    "img_src": "http://example.com/image1.jpg",
                    "url": "http://example.com",
                    "title": "Example Image 1"
                }
            ]
        })
        
        with patch("agents.image_search_agent.search_searxng", new_callable=AsyncMock) as mock_search, \
            patch.object(image_search_agent, "create_retriever_chain", return_value=mock_chain):
            mock_search.return_value = {
                "results": [
                    {
                        "img_src": "http://example.com/image1.jpg",
                        "url": "http://example.com",
                        "title": "Example Image 1"
                    }
                ]
            }
            
            result = await mock_chain.invoke({
                "query": "test query",
                "chat_history": []
            })
            
            assert result["query"] == "rephrased query"
            assert len(result["images"]) == 1
            assert all(key in img for img in result["images"] for key in ["img_src", "url", "title"])

    @pytest.mark.asyncio
    async def test_process_output_no_results(self, image_search_agent):
        """Test processing when no results are found."""
        mock_chain = AsyncMock()
        mock_chain.invoke = AsyncMock(return_value={
            "query": "rephrased query",
            "images": []
        })
        
        with patch("agents.image_search_agent.search_searxng", new_callable=AsyncMock) as mock_search, \
            patch.object(image_search_agent, "create_retriever_chain", return_value=mock_chain):
            mock_search.return_value = {"results": []}
            
            result = await mock_chain.invoke({
                "query": "test query",
                "chat_history": []
            })
            
            assert result["query"] == "rephrased query"
            assert len(result["images"]) == 0

    @pytest.mark.asyncio
    async def test_create_answering_chain(self, image_search_agent):
        """Test the creation of the answering chain."""
        mock_chain = AsyncMock()
        with patch.object(image_search_agent, "create_answering_chain", return_value=mock_chain):
            chain = await image_search_agent.create_answering_chain()
            assert chain is not None

    @pytest.mark.asyncio
    async def test_generate_response_success(self, image_search_agent):
        """Test successful response generation."""
        mock_chain = AsyncMock()
        mock_chain.invoke = AsyncMock(return_value={
            "type": "images",
            "data": [{
                "img_src": "http://example.com/image.jpg",
                "url": "http://example.com",
                "title": "Example Image"
            }]
        })
        
        with patch.object(image_search_agent, "create_answering_chain", return_value=mock_chain):
            response = await mock_chain.invoke({
                "query": "test query",
                "chat_history": []
            })
            
            assert response["type"] == "images"
            assert len(response["data"]) == 1

    @pytest.mark.asyncio
    async def test_generate_response_no_images(self, image_search_agent):
        """Test response generation when no images are found."""
        mock_chain = AsyncMock()
        mock_chain.invoke = AsyncMock(return_value={
            "type": "error",
            "data": "No images found. Please try a different search query."
        })
        
        with patch.object(image_search_agent, "create_answering_chain", return_value=mock_chain):
            response = await mock_chain.invoke({
                "query": "test query",
                "chat_history": []
            })
            
            assert response["type"] == "error"
            assert "No images found" in response["data"]

    @pytest.mark.asyncio
    async def test_handle_image_search(self, image_search_agent):
        """Test the handle_image_search function."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_embeddings = MagicMock(spec=Embeddings)
        
        with patch("agents.image_search_agent.ImageSearchAgent", return_value=image_search_agent):
            async for result in handle_image_search(
                query="test query",
                history=[],
                llm=mock_llm,
                embeddings=mock_embeddings
            ):
                assert isinstance(result, dict)
                assert "type" in result
                assert "data" in result

    def test_image_search_agent_inheritance(self):
        """Test that ImageSearchAgent inherits from AbstractAgent."""
        assert issubclass(ImageSearchAgent, AbstractAgent)

    @pytest.mark.asyncio
    async def test_error_handling(self, image_search_agent):
        """Test error handling in the retriever chain."""
        mock_chain = AsyncMock()
        mock_chain.invoke = AsyncMock(return_value={
            "query": "",
            "images": []
        })
        
        with patch("agents.image_search_agent.search_searxng", new_callable=AsyncMock) as mock_search, \
            patch.object(image_search_agent, "create_retriever_chain", return_value=mock_chain):
            mock_search.side_effect = Exception("Test error")
            
            result = await mock_chain.invoke({
                "query": "test query",
                "chat_history": []
            })
            
            assert result["query"] == ""
            assert len(result["images"]) == 0

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_rate_limit_error(self, image_search_agent):
        """Test rate limit handling in retriever chain"""
        with patch("agents.image_search_agent.search_searxng",
                AsyncMock(side_effect=Exception("Rate limit exceeded"))), \
            patch.object(logger, "error") as mock_logger:
            
            chain = await image_search_agent.create_retriever_chain()
            result = await chain.ainvoke({
                "query": "test",
                "chat_history": []
            })
            
            assert "rate limit" in result["response"].lower()
            mock_logger.assert_called_with("Error in image search: %s", "Rate limit exceeded")

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_different_configurations(self, mock_llm, mock_embeddings):
        """Test retriever chain with different configurations"""
        for config in ["default", "strict", "lenient"]:
            agent = ImageSearchAgent(mock_llm, mock_embeddings, config)
            chain = await agent.create_retriever_chain()
            assert chain is not None

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_large_input(self, image_search_agent):
        """Test retriever chain with large input"""
        large_query = "a" * 10000
        chain = await image_search_agent.create_retriever_chain()
        result = await chain.ainvoke({
            "query": large_query,
            "chat_history": []
        })
        assert len(result["query"]) < len(large_query)  # Should be processed/truncated

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_special_characters(self, image_search_agent):
        """Test retriever chain with special characters"""
        special_query = "!@#$%^&*()"
        chain = await image_search_agent.create_retriever_chain()
        result = await chain.ainvoke({
            "query": special_query,
            "chat_history": []
        })
        assert result["query"] != special_query  # Should be processed

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_empty_query(self, image_search_agent):
        """Test retriever chain with empty query"""
        chain = await image_search_agent.create_retriever_chain()
        result = await chain.ainvoke({
            "query": "",
            "chat_history": []
        })
        assert result["query"] == ""
        assert len(result["images"]) == 0
