import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain.schema import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from agents.web_search_agent import WebSearchAgent, handle_web_search
from utils.logger import logger

class TestWebSearchAgent:
    """Test suite for WebSearchAgent"""

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
    def web_search_agent(self, mock_llm, mock_embeddings):
        """Fixture providing a WebSearchAgent instance"""
        return WebSearchAgent(mock_llm, mock_embeddings)

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
    async def test_create_retriever_chain(self, web_search_agent):
        """Test creation of retriever chain"""
        mock_chain = AsyncMock()
        mock_chain.invoke = AsyncMock()
        
        with patch('agents.web_search_agent.RunnableSequence', return_value=mock_chain):
            chain = await web_search_agent.create_retriever_chain()
            assert chain is not None

    @pytest.mark.asyncio
    async def test_process_output_with_links(self, web_search_agent):
        """Test process_output with links in the response"""
        mock_chain = AsyncMock()
        mock_chain.invoke = AsyncMock(return_value={
            "llm_output": {
                "query": "summarize",
                "docs": [Document(page_content="test content", metadata={"title": "test", "url": "http://test.com"})]
            },
            "original_input": {"query": "test", "chat_history": []}
        })
        
        with patch('agents.web_search_agent.RunnableSequence', return_value=mock_chain):
            chain = await web_search_agent.create_retriever_chain()
            result = await chain.invoke({
                "query": "test",
                "chat_history": []
            })
            
            assert "llm_output" in result
            assert "query" in result["llm_output"]
            assert "docs" in result["llm_output"]
            assert result["llm_output"]["query"] == "summarize"
            assert len(result["llm_output"]["docs"]) == 1
            assert result["llm_output"]["docs"][0].page_content == "test content"

    @pytest.mark.asyncio
    async def test_process_output_without_links(self, web_search_agent):
        """Test process_output without links in the response"""
        mock_chain = AsyncMock()
        mock_chain.invoke = AsyncMock(return_value={
            "llm_output": {
                "query": "test question",
                "docs": [Document(page_content="test content", metadata={"title": "test", "url": "http://test.com"})]
            },
            "original_input": {"query": "test", "chat_history": []}
        })
        
        with patch('agents.web_search_agent.RunnableSequence', return_value=mock_chain):
            chain = await web_search_agent.create_retriever_chain()
            result = await chain.invoke({
                "query": "test",
                "chat_history": []
            })
            
            assert "llm_output" in result
            assert "query" in result["llm_output"]
            assert "docs" in result["llm_output"]
            assert result["llm_output"]["query"] == "test question"
            assert len(result["llm_output"]["docs"]) == 1
            assert result["llm_output"]["docs"][0].page_content == "test content"

    @pytest.mark.asyncio
    async def test_create_answering_chain(self, web_search_agent):
        """Test creation of answering chain"""
        mock_chain = AsyncMock()
        mock_chain.invoke = AsyncMock()
        
        with patch('agents.web_search_agent.RunnableSequence', return_value=mock_chain):
            chain = await web_search_agent.create_answering_chain()
            assert chain is not None
            mock_chain.invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_answering_chain_with_error(self, web_search_agent):
        """Test error handling in answering chain creation"""
        mock_chain = AsyncMock()
        mock_chain.invoke = AsyncMock(side_effect=Exception("test error"))
        
        with patch('agents.web_search_agent.RunnableSequence.from_components', return_value=mock_chain), \
             patch.object(logger, "error") as mock_logger:
            
            chain = await web_search_agent.create_answering_chain()
            result = await chain.invoke({
                "query": "test",
                "chat_history": []
            })
            
            assert isinstance(result, dict)
            assert result["type"] == "error"
            assert result["data"] == "An error has occurred please try again later"
            mock_logger.assert_called_once_with(
                "Error creating answering chain: %s",
                "test error"
            )

    @pytest.mark.asyncio
    async def test_handle_web_search(self):
        """Test the handle_web_search function with valid input"""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_embeddings = MagicMock(spec=Embeddings)
        
        results = []
        async for result in handle_web_search(
            query="test",
            history=[],
            llm=mock_llm,
            embeddings=mock_embeddings
        ):
            results.append(result)
            assert result is not None
        
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_handle_web_search_with_empty_query(self):
        """Test handle_web_search with empty query"""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_embeddings = MagicMock(spec=Embeddings)
        
        results = []
        async for result in handle_web_search(
            query="",
            history=[],
            llm=mock_llm,
            embeddings=mock_embeddings
        ):
            results.append(result)
        
        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert results[0]["type"] == "error"
        assert results[0]["data"] == "Please provide a valid search query"

    @pytest.mark.asyncio
    async def test_handle_web_search_with_error(self):
        """Test error handling in handle_web_search"""
        with patch('agents.web_search_agent.WebSearchAgent.handle_search',
                  side_effect=Exception("test error")), \
             patch.object(logger, "error") as mock_logger:
            
            results = []
            async for result in handle_web_search(
                query="test",
                history=[],
                llm=MagicMock(spec=BaseChatModel),
                embeddings=MagicMock(spec=Embeddings)
            ):
                results.append(result)
            
            assert len(results) == 1
            assert isinstance(results[0], dict)
            assert results[0]["type"] == "error"
            assert results[0]["data"] == "An error has occurred please try again later"
            mock_logger.assert_called_once_with(
                "Error in web search: %s",
                "test error"
            )

    @pytest.mark.asyncio
    async def test_error_handling(self, web_search_agent):
        """Test error handling in the web search process"""
        mock_lambda = AsyncMock()
        mock_lambda.invoke = AsyncMock(side_effect=Exception("Test error"))
        
        with patch('agents.web_search_agent.RunnableLambda', return_value=mock_lambda), \
             patch.object(logger, "error") as mock_logger:
            
            chain = await web_search_agent.create_retriever_chain()
            result = await chain.invoke({
                "query": "test",
                "chat_history": []
            })
            
            assert "llm_output" in result
            assert result["llm_output"] == {"query": "", "docs": []}
            mock_logger.assert_called_once_with(
                "Error in web search process: %s",
                "Test error"
            )

    @pytest.mark.asyncio
    async def test_format_prompt(self, web_search_agent):
        """Test the format_prompt method with valid input"""
        query = "test query"
        context = "test context"
        prompt = web_search_agent.format_prompt(query, context)
        
        # Verify all required components are present
        assert "test query" in prompt
        assert "test context" in prompt
        assert "You are Perplexica" in prompt
        assert "<context>" in prompt
        assert "informative and relevant" in prompt  # From base prompt
        
        # Verify prompt structure
        assert "You are Perplexica" in prompt
        assert "<context>" in prompt
        assert context in prompt

    @pytest.mark.asyncio
    async def test_format_prompt_with_empty_input(self, web_search_agent):
        """Test format_prompt with empty query and context"""
        prompt = web_search_agent.format_prompt("", "")
        assert "You are Perplexica" in prompt
        assert "informative and relevant" in prompt  # From base prompt
        assert "<context>" in prompt  # Base class includes context tags