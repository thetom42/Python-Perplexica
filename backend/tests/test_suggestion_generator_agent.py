"""
Unit tests for the Suggestion Generator Agent Module
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableSequence
from agents.suggestion_generator_agent import (
    SuggestionGeneratorAgent,
    handle_suggestion_generation
)
from utils.logger import logger

class TestSuggestionGeneratorAgent:
    """Test suite for SuggestionGeneratorAgent"""

    @pytest.fixture
    def mock_llm(self):
        """Fixture providing a mock LLM instance"""
        llm = MagicMock(spec=BaseChatModel)
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="<suggestions>\nTest suggestion 1\nTest suggestion 2\n</suggestions>"))
        return llm

    @pytest.fixture
    def mock_embeddings(self):
        """Fixture providing a mock embeddings instance"""
        return MagicMock(spec=Embeddings)

    @pytest.fixture
    def suggestion_generator_agent(self, mock_llm, mock_embeddings):
        """Fixture providing a SuggestionGeneratorAgent instance"""
        return SuggestionGeneratorAgent(mock_llm, mock_embeddings)

    @pytest.fixture
    def chat_history(self) -> List[BaseMessage]:
        """Fixture providing standard chat history"""
        return [
            HumanMessage(content="What is the capital of France?"),
            AIMessage(content="The capital of France is Paris.")
        ]

    @pytest.fixture
    def empty_chat_history(self) -> List[BaseMessage]:
        """Fixture providing empty chat history"""
        return []

    @pytest.mark.asyncio
    async def test_handle_suggestion_generation_success(self, suggestion_generator_agent, chat_history):
        """Test successful suggestion generation with valid input"""
        # Create an async generator function for the mock
        async def mock_search_gen(query: str, history: List[BaseMessage]):
            yield {"type": "response", "data": ["Test suggestion 1", "Test suggestion 2"]}

        # Mock the entire agent behavior
        mock_agent = MagicMock(spec=SuggestionGeneratorAgent)
        mock_agent.handle_search = mock_search_gen
        
        # Mock the agent creation to return our mock agent
        with patch('agents.suggestion_generator_agent.SuggestionGeneratorAgent', return_value=mock_agent):
            suggestions = await handle_suggestion_generation(chat_history, suggestion_generator_agent.llm, suggestion_generator_agent.embeddings)
            assert isinstance(suggestions, list)
            assert len(suggestions) == 2
            assert "Test suggestion 1" in suggestions
            assert "Test suggestion 2" in suggestions

    @pytest.mark.asyncio
    async def test_handle_suggestion_generation_empty_history(self, suggestion_generator_agent, empty_chat_history):
        """Test suggestion generation with empty chat history"""
        suggestions = await handle_suggestion_generation(empty_chat_history, suggestion_generator_agent.llm, suggestion_generator_agent.embeddings)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0  # Should still generate suggestions even with empty history

    @pytest.mark.asyncio
    async def test_handle_suggestion_generation_error(self, suggestion_generator_agent, chat_history):
        """Test error handling in suggestion generation"""
        # Create a mock agent that raises an exception
        mock_agent = AsyncMock(spec=SuggestionGeneratorAgent)
        
        async def mock_handle_search(*args, **kwargs):
            raise Exception("Test error")
            yield  # This will never be reached, but needed for async generator syntax
        
        # Set up the mock agent's handle_search method
        mock_agent.handle_search = mock_handle_search
        
        # Mock both the agent creation and logger
        with patch('agents.suggestion_generator_agent.SuggestionGeneratorAgent', return_value=mock_agent), \
             patch('agents.suggestion_generator_agent.logger.error') as mock_error:
            
            suggestions = await handle_suggestion_generation(chat_history, suggestion_generator_agent.llm, suggestion_generator_agent.embeddings)
            
            # Verify the error was logged correctly
            mock_error.assert_called_once_with("Error in suggestion generation: %s", "Test error")
            
            # Verify the error response
            assert isinstance(suggestions, list)
            assert len(suggestions) == 1
            assert "Unable to generate suggestions" in suggestions[0]

    @pytest.mark.asyncio
    async def test_suggestion_generator_agent_create_answering_chain(self, suggestion_generator_agent):
        """Test creation of answering chain"""
        chain = await suggestion_generator_agent.create_answering_chain()
        assert isinstance(chain, RunnableSequence)

    @pytest.mark.asyncio
    async def test_suggestion_generator_agent_create_retriever_chain(self, suggestion_generator_agent):
        """Test creation of retriever chain"""
        chain = await suggestion_generator_agent.create_retriever_chain()
        assert isinstance(chain, RunnableSequence)

    @pytest.mark.asyncio
    async def test_suggestion_generator_agent_parse_response(self, suggestion_generator_agent):
        """Test response parsing"""
        response = "Suggestion 1\nSuggestion 2"
        parsed = await suggestion_generator_agent.parse_response(response)
        assert parsed["type"] == "response"
        assert isinstance(parsed["data"], list)
        assert len(parsed["data"]) == 2

    @pytest.mark.asyncio
    async def test_suggestion_generator_agent_parse_empty_response(self, suggestion_generator_agent):
        """Test parsing of empty response"""
        response = ""
        parsed = await suggestion_generator_agent.parse_response(response)
        assert parsed["type"] == "response"
        assert isinstance(parsed["data"], list)
        assert len(parsed["data"]) == 0

    @pytest.mark.asyncio
    async def test_suggestion_generator_agent_parse_invalid_response(self, suggestion_generator_agent):
        """Test parsing of invalid response format"""
        response = "Invalid format"
        with pytest.raises(ValueError):
            await suggestion_generator_agent.parse_response(response)

    @pytest.mark.asyncio
    async def test_suggestion_generator_agent_with_invalid_llm(self):
        """Test initialization with invalid LLM type"""
        with pytest.raises(TypeError):
            SuggestionGeneratorAgent(None, MagicMock(spec=Embeddings))

    @pytest.mark.asyncio
    async def test_suggestion_generator_agent_with_invalid_embeddings(self, mock_llm):
        """Test initialization with invalid embeddings type"""
        with pytest.raises(TypeError):
            SuggestionGeneratorAgent(mock_llm, None)

    @pytest.mark.asyncio
    async def test_suggestion_generator_agent_with_rate_limit_error(self, suggestion_generator_agent, chat_history):
        """Test rate limit handling in suggestion generation"""
        # Create a mock chain that raises a rate limit error
        async def mock_search_gen(query: str, history: List[BaseMessage]):
            raise Exception("Rate limit exceeded")
            yield  # This will never be reached

        # Mock the agent's handle_search method
        mock_agent = MagicMock(spec=SuggestionGeneratorAgent)
        mock_agent.handle_search = mock_search_gen
        
        # Mock the agent creation
        with patch('agents.suggestion_generator_agent.SuggestionGeneratorAgent', return_value=mock_agent), \
             patch.object(logger, "error") as mock_logger:
            suggestions = await handle_suggestion_generation(chat_history, suggestion_generator_agent.llm, suggestion_generator_agent.embeddings)
            assert isinstance(suggestions, list)
            assert len(suggestions) == 1
            assert "rate limit" in suggestions[0].lower()
            mock_logger.assert_called_once_with("Error in suggestion generation: %s", "Rate limit exceeded")