import pytest
from unittest.mock import AsyncMock, patch
from langchain.schema import HumanMessage, AIMessage
from backend.agents.writing_assistant_agent import handle_writing_assistance

@pytest.fixture
def mock_chat_model():
    return AsyncMock()

@pytest.fixture
def mock_embeddings_model():
    return AsyncMock()

@pytest.mark.asyncio
async def test_handle_writing_assistance(mock_chat_model, mock_embeddings_model):
    query = "Help me write an introduction for an essay about climate change"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I assist you with your writing today?")
    ]

    with patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_chat_model.arun.return_value = "Here's a possible introduction for your essay on climate change:\n\nClimate change is one of the most pressing issues facing our planet today. As global temperatures continue to rise, we are witnessing unprecedented changes in weather patterns, ecosystems, and human societies. This essay will explore the causes and consequences of climate change, as well as potential solutions to mitigate its effects. By understanding the science behind climate change and its far-reaching impacts, we can work towards a more sustainable future for generations to come."

        result = await handle_writing_assistance(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "climate change" in result["data"].lower()
        assert "global temperatures" in result["data"].lower()
        assert "sustainable future" in result["data"].lower()
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_writing_assistance_grammar_check(mock_chat_model, mock_embeddings_model):
    query = "Can you check this sentence for grammar errors? 'The cat were sleeping on the couch.'"
    history = []

    with patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_chat_model.arun.return_value = "The sentence has a grammar error. The correct version should be:\n\n'The cat was sleeping on the couch.'\n\nThe error was in the verb 'were'. Since 'cat' is singular, we should use the singular form of the verb 'to be', which is 'was' in this case."

        result = await handle_writing_assistance(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "grammar error" in result["data"].lower()
        assert "The cat was sleeping on the couch" in result["data"]
        assert "singular" in result["data"].lower()
        mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_handle_writing_assistance_paraphrasing(mock_chat_model, mock_embeddings_model):
    query = "Can you help me paraphrase this sentence? 'The quick brown fox jumps over the lazy dog.'"
    history = []

    with patch('backend.lib.providers.get_chat_model', return_value=mock_chat_model), \
         patch('backend.lib.providers.get_embeddings_model', return_value=mock_embeddings_model):

        mock_chat_model.arun.return_value = "Here are a few ways to paraphrase the sentence:\n\n1. 'The agile auburn fox leaps above the idle canine.'\n2. 'A swift russet fox hops over a sluggish hound.'\n3. 'The nimble tawny fox bounds across the lethargic dog.'\n\nEach of these paraphrases maintains the meaning of the original sentence while using different vocabulary and sentence structures."

        result = await handle_writing_assistance(query, history, mock_chat_model, mock_embeddings_model)

        assert result["type"] == "response"
        assert "paraphrase" in result["data"].lower()
        assert "fox" in result["data"].lower()
        assert "dog" in result["data"].lower()
        assert "meaning" in result["data"].lower()
        mock_chat_model.arun.assert_called_once()
