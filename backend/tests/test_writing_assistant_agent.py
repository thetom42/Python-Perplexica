import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, AsyncGenerator
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from agents.writing_assistant_agent import WritingAssistantAgent, handle_writing_assistance
from utils.logger import logger

class TestWritingAssistantAgent:
    """Test suite for WritingAssistantAgent"""

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
    def writing_assistant(self, mock_llm, mock_embeddings):
        """Fixture providing a WritingAssistantAgent instance"""
        return WritingAssistantAgent(mock_llm, mock_embeddings, "balanced")

    @pytest.fixture
    def mock_generation(self):
        """Fixture providing a mock Generation object"""
        from langchain.schema import Generation
        return Generation(text="test response")

    @pytest.fixture
    def mock_chat_history(self):
        """Fixture providing standard chat history"""
        return [HumanMessage(content="previous message")]

    @pytest.mark.asyncio
    async def test_create_retriever_chain(self, writing_assistant):
        """Test retriever chain creation with valid query"""
        chain = await writing_assistant.create_retriever_chain()
        result = await chain.ainvoke({"query": "test query"})
        assert result == {"query": "test query", "docs": []}

    @pytest.mark.asyncio
    async def test_create_answering_chain_success(self, writing_assistant, mock_llm, mock_generation):
        """Test answering chain with successful response"""
        mock_llm.ainvoke.return_value = mock_generation
        
        chain = await writing_assistant.create_answering_chain()
        result = await chain.ainvoke({
            "query": "test query",
            "chat_history": []
        })
        
        assert result["type"] == "response"
        assert result["data"] == "test response"

    @pytest.mark.asyncio
    async def test_create_answering_chain_error(self, writing_assistant, mock_llm):
        """Test error handling in answering chain"""
        mock_llm.ainvoke.side_effect = Exception("Test error")
        
        chain = await writing_assistant.create_answering_chain()
        result = await chain.ainvoke({
            "query": "test query",
            "chat_history": []
        })
        
        assert result["type"] == "error"
        assert "error occurred" in result["data"]

    @pytest.mark.asyncio
    async def test_create_answering_chain_rate_limit(self, writing_assistant, mock_llm):
        """Test rate limit handling in answering chain"""
        mock_llm.ainvoke.side_effect = Exception("Rate limit exceeded")
        
        chain = await writing_assistant.create_answering_chain()
        result = await chain.ainvoke({
            "query": "test query",
            "chat_history": []
        })
        
        assert result["type"] == "error"
        assert "rate limit" in result["data"].lower()

    def test_format_prompt(self, writing_assistant):
        """Test prompt formatting"""
        prompt = writing_assistant.format_prompt("test query", "")
        assert "Writing Assistant" in prompt
        assert "helping the user with their writing tasks" in prompt

    @pytest.mark.asyncio
    async def test_parse_response(self, writing_assistant):
        """Test response parsing"""
        response = "test response"
        result = await writing_assistant.parse_response(response)
        assert result["type"] == "response"
        assert result["data"] == response

    @pytest.mark.asyncio
    async def test_parse_response_empty(self, writing_assistant):
        """Test parsing empty response"""
        result = await writing_assistant.parse_response("")
        assert result["type"] == "response"
        assert result["data"] == ""

    @pytest.mark.asyncio
    async def test_handle_writing_assistance_success(self, mock_llm, mock_generation, mock_chat_history):
        """Test writing assistance handler with successful response"""
        mock_llm.ainvoke.return_value = mock_generation
        
        results = []
        async for result in handle_writing_assistance("test query", mock_chat_history, mock_llm):
            results.append(result)
        
        assert len(results) > 0
        assert results[0]["type"] == "response"
        assert "test response" in results[0]["data"]

    @pytest.mark.asyncio
    async def test_handle_writing_assistance_error(self, mock_llm, mock_chat_history):
        """Test error handling in writing assistance handler"""
        mock_llm.ainvoke.side_effect = Exception("Test error")
        
        results = []
        async for result in handle_writing_assistance("test query", mock_chat_history, mock_llm):
            results.append(result)
        
        assert len(results) > 0
        assert results[0]["type"] == "error"
        assert "error occurred" in results[0]["data"]

    @pytest.mark.asyncio
    async def test_handle_writing_assistance_empty_query(self, mock_llm, mock_chat_history):
        """Test handling empty query in writing assistance"""
        results = []
        async for result in handle_writing_assistance("", mock_chat_history, mock_llm):
            results.append(result)
        
        assert len(results) > 0
        assert "Please provide a query" in results[0]["data"]

    @pytest.mark.asyncio
    async def test_handle_writing_assistance_invalid_history(self, mock_llm):
        """Test handling invalid chat history"""
        with pytest.raises(TypeError):
            async for _ in handle_writing_assistance("test query", "invalid", mock_llm):
                pass

    @pytest.mark.asyncio
    async def test_handle_writing_assistance_logging(self, mock_llm, mock_chat_history):
        """Test logging in writing assistance handler"""
        with patch.object(logger, "error") as mock_logger:
            mock_llm.ainvoke.side_effect = Exception("Test error")
            
            async for _ in handle_writing_assistance("test query", mock_chat_history, mock_llm):
                pass
            
            mock_logger.assert_called_once_with("Error in writing assistance: %s", "Test error")
