import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from agents.video_search_agent import VideoSearchAgent, handle_video_search
from utils.logger import logger

class TestVideoSearchAgent:
    """Test suite for VideoSearchAgent"""

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
    def video_search_agent(self, mock_llm, mock_embeddings):
        """Fixture providing a VideoSearchAgent instance"""
        return VideoSearchAgent(mock_llm, mock_embeddings, "balanced")

    @pytest.fixture
    def mock_video_results(self):
        """Fixture providing standard video search results"""
        return [{
            "title": "test video",
            "description": "test description",
            "url": "http://test.com/video",
            "thumbnail": "http://test.com/thumbnail.jpg",
            "duration": "10:00"
        }]

    @pytest.fixture
    def mock_empty_results(self):
        """Fixture providing empty search results"""
        return []

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_valid_query(self, video_search_agent):
        """Test retriever chain creation with valid query"""
        chain = await video_search_agent.create_retriever_chain()
        assert chain is not None

    @pytest.mark.asyncio
    async def test_create_retriever_chain_with_error(self, video_search_agent):
        """Test error handling in retriever chain creation"""
        with patch("agents.video_search_agent.VideoSearchAgent.create_retriever_chain",
                 AsyncMock(side_effect=Exception("test error"))), \
             patch.object(logger, "error") as mock_logger:
            
            async for result in video_search_agent.basic_search("test query", []):
                assert result["type"] == "error"
                assert "test error" in result["data"]
            mock_logger.assert_called_once_with("Error in search: %s", "test error")

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_valid_response(self, video_search_agent):
        """Test answering chain creation with valid response"""
        chain = await video_search_agent.create_answering_chain()
        assert chain is not None

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_error(self, video_search_agent):
        """Test error handling in answering chain creation"""
        with patch("agents.video_search_agent.VideoSearchAgent.create_answering_chain",
                 AsyncMock(side_effect=Exception("test error"))), \
             patch.object(logger, "error") as mock_logger:
            
            async for result in video_search_agent.basic_search("test query", []):
                assert result["type"] == "error"
                assert "test error" in result["data"]
            mock_logger.assert_called_once_with("Error in search: %s", "test error")

    def test_format_prompt_with_valid_input(self, video_search_agent):
        """Test prompt formatting with valid input"""
        query = "test query"
        context = "test context"
        prompt = video_search_agent.format_prompt(query, context)
        assert isinstance(prompt, str)
        assert query in prompt
        assert context in prompt

    def test_format_prompt_with_empty_input(self, video_search_agent):
        """Test prompt formatting with empty input"""
        with pytest.raises(ValueError):
            video_search_agent.format_prompt("", "")

    @pytest.mark.asyncio
    async def test_parse_response_with_valid_response(self, video_search_agent):
        """Test response parsing with valid response"""
        response = "test response"
        result = await video_search_agent.parse_response(response)
        assert isinstance(result, dict)
        assert result["type"] == "response"
        assert result["data"] == response

    @pytest.mark.asyncio
    async def test_parse_response_with_error_response(self, video_search_agent):
        """Test response parsing with error response"""
        response = "error: test error"
        result = await video_search_agent.parse_response(response)
        assert isinstance(result, dict)
        assert result["type"] == "error"
        assert "test error" in result["data"]

    @pytest.mark.asyncio
    @patch("agents.video_search_agent.VideoSearchAgent")
    async def test_handle_video_search_with_valid_results(self, mock_agent, mock_llm, mock_embeddings, mock_video_results):
        """Test video search handling with valid results"""
        mock_instance = mock_agent.return_value
        
        class MockAsyncIterator:
            def __init__(self):
                self.data = [{"type": "response", "data": mock_video_results}]
            
            def __aiter__(self):
                return self
                
            async def __anext__(self):
                if self.data:
                    return self.data.pop(0)
                raise StopAsyncIteration
        
        mock_instance.handle_search = lambda *args, **kwargs: MockAsyncIterator()
        
        results = await handle_video_search("test query", [], mock_llm, mock_embeddings)
        assert len(results) == 1
        assert results[0]["title"] == "test video"

    @pytest.mark.asyncio
    @patch("agents.video_search_agent.VideoSearchAgent")
    async def test_handle_video_search_with_empty_results(self, mock_agent, mock_llm, mock_embeddings, mock_empty_results):
        """Test video search handling with empty results"""
        mock_instance = mock_agent.return_value
        
        class MockAsyncIterator:
            def __init__(self):
                self.data = [{"type": "response", "data": mock_empty_results}]
            
            def __aiter__(self):
                return self
                
            async def __anext__(self):
                if self.data:
                    return self.data.pop(0)
                raise StopAsyncIteration
        
        mock_instance.handle_search = lambda *args, **kwargs: MockAsyncIterator()
        
        results = await handle_video_search("test query", [], mock_llm, mock_embeddings)
        assert len(results) == 0

    @pytest.mark.asyncio
    @patch("agents.video_search_agent.VideoSearchAgent")
    async def test_handle_video_search_with_error(self, mock_agent, mock_llm, mock_embeddings):
        """Test error handling in video search"""
        mock_instance = mock_agent.return_value
        
        class MockAsyncIterator:
            def __init__(self):
                self.data = [{"type": "error", "data": "test error"}]
            
            def __aiter__(self):
                return self
                
            async def __anext__(self):
                if self.data:
                    return self.data.pop(0)
                raise StopAsyncIteration
        
        mock_instance.handle_search = lambda *args, **kwargs: MockAsyncIterator()
        
        with patch.object(logger, "error") as mock_logger:
            results = await handle_video_search("test query", [], mock_llm, mock_embeddings)
            assert len(results) == 0
            mock_logger.assert_called_once_with("Error in video search: %s", "test error")

    @pytest.mark.asyncio
    async def test_handle_video_search_with_invalid_input(self, mock_llm, mock_embeddings):
        """Test video search with invalid input types"""
        with pytest.raises(TypeError):
            await handle_video_search(None, "invalid", mock_llm, mock_embeddings)