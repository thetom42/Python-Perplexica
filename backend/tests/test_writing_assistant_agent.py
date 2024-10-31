import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain.schema import HumanMessage, AIMessage
from backend.agents.writing_assistant_agent import (
    handle_writing_assistant,
    create_writing_assistant_chain,
    RunnableSequence
)

@pytest.fixture
def mock_chat_model():
    mock = AsyncMock()
    mock.arun = AsyncMock()
    return mock

@pytest.fixture
def mock_embeddings_model():
    return AsyncMock()

@pytest.mark.asyncio
async def test_handle_writing_assistance_basic(mock_chat_model):
    query = "Help me write an introduction for an essay about climate change"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!")
    ]

    mock_chat_model.arun.return_value = """
    Here's a possible introduction for your essay on climate change:

    Climate change is one of the most pressing issues facing our planet today. As global temperatures 
    continue to rise, we are witnessing unprecedented changes in weather patterns, ecosystems, and 
    human societies. This essay will explore the causes and consequences of climate change, as well 
    as potential solutions to mitigate its effects.
    """

    async for result in handle_writing_assistant(query, history, mock_chat_model):
        assert result["type"] == "response"
        assert "climate change" in result["data"].lower()
        assert "global temperatures" in result["data"].lower()

@pytest.mark.asyncio
async def test_writing_assistant_chain_creation(mock_chat_model):
    chain = await create_writing_assistant_chain(mock_chat_model)
    assert isinstance(chain, RunnableSequence)
    
    mock_chat_model.arun.return_value = "Writing assistance response"
    
    result = await chain.invoke({
        "query": "Help me write something",
        "chat_history": []
    })
    
    assert result["type"] == "response"
    assert isinstance(result["data"], str)

@pytest.mark.asyncio
async def test_error_handling(mock_chat_model):
    query = "test query"
    history = []

    mock_chat_model.arun.side_effect = Exception("LLM error")
    
    async for result in handle_writing_assistant(query, history, mock_chat_model):
        assert result["type"] == "error"
        assert "error" in result["data"].lower()

@pytest.mark.asyncio
async def test_grammar_correction(mock_chat_model):
    query = "Check grammar: 'The cat were sleeping on the couch.'"
    history = []

    mock_chat_model.arun.return_value = """
    The sentence has a grammar error. The correct version is:
    'The cat was sleeping on the couch.'

    Explanation:
    - 'Cat' is singular, so it needs the singular verb form 'was'
    - 'Were' is used for plural subjects
    """

    async for result in handle_writing_assistant(query, history, mock_chat_model):
        assert result["type"] == "response"
        assert "was sleeping" in result["data"].lower()
        assert "explanation" in result["data"].lower()

@pytest.mark.asyncio
async def test_paraphrasing_assistance(mock_chat_model):
    query = "Help me paraphrase: 'The quick brown fox jumps over the lazy dog.'"
    history = []

    mock_chat_model.arun.return_value = """
    Here are some paraphrased versions:
    1. A swift russet fox leaps above the idle canine.
    2. The agile brown fox bounds over the sleepy hound.
    3. A nimble tawny fox hops across the lethargic dog.
    """

    async for result in handle_writing_assistant(query, history, mock_chat_model):
        assert result["type"] == "response"
        assert "fox" in result["data"].lower()
        assert "dog" in result["data"].lower()

@pytest.mark.asyncio
async def test_writing_style_suggestions(mock_chat_model):
    query = "How can I make this sentence more formal: 'This thing is really good.'"
    history = []

    mock_chat_model.arun.return_value = """
    Here are more formal alternatives:
    1. "This product demonstrates exceptional quality."
    2. "This item exhibits remarkable effectiveness."
    3. "This solution proves highly beneficial."

    The improvements include:
    - Using specific nouns instead of "thing"
    - Replacing informal intensifiers like "really"
    - Employing more sophisticated vocabulary
    """

    async for result in handle_writing_assistant(query, history, mock_chat_model):
        assert result["type"] == "response"
        assert "formal" in result["data"].lower()
        assert "improvements" in result["data"].lower()

@pytest.mark.asyncio
async def test_chat_history_integration(mock_chat_model):
    query = "Can you revise it to be more concise?"
    history = [
        HumanMessage(content="Help me write about renewable energy"),
        AIMessage(content="Renewable energy sources like solar and wind power are sustainable alternatives to fossil fuels.")
    ]

    mock_chat_model.arun.return_value = "Renewable energy offers sustainable alternatives to fossil fuels."

    async for result in handle_writing_assistant(query, history, mock_chat_model):
        assert result["type"] == "response"
        assert "renewable energy" in result["data"].lower()
        
        # Verify chat history was passed to the model
        call_args = mock_chat_model.arun.call_args[1]
        assert "chat_history" in call_args
        assert any("renewable energy" in str(msg.content) for msg in call_args["chat_history"])

@pytest.mark.asyncio
async def test_unused_embeddings_parameter(mock_chat_model, mock_embeddings_model):
    query = "Help me write something"
    history = []

    # Should work the same with or without embeddings
    async for result in handle_writing_assistant(query, history, mock_chat_model, mock_embeddings_model):
        assert result["type"] in ["response", "error"]
    
    async for result in handle_writing_assistant(query, history, mock_chat_model, None):
        assert result["type"] in ["response", "error"]

@pytest.mark.asyncio
async def test_empty_query_handling(mock_chat_model):
    query = ""
    history = []

    mock_chat_model.arun.return_value = "I'm not sure what you'd like help writing. Could you please provide more details?"

    async for result in handle_writing_assistant(query, history, mock_chat_model):
        assert result["type"] == "response"
        assert "provide more details" in result["data"].lower()

@pytest.mark.asyncio
async def test_long_content_handling(mock_chat_model):
    query = "Help me write a long essay about artificial intelligence"
    history = []

    # Simulate a long response
    mock_chat_model.arun.return_value = "A" * 10000 + "\nB" * 10000

    async for result in handle_writing_assistant(query, history, mock_chat_model):
        assert result["type"] == "response"
        assert len(result["data"]) > 15000  # Should handle long content

@pytest.mark.asyncio
async def test_invalid_history_handling(mock_chat_model):
    query = "Help me write something"
    invalid_history = ["not a valid message"]  # Invalid history format

    async for result in handle_writing_assistant(query, invalid_history, mock_chat_model):
        # Should handle invalid history gracefully
        assert result["type"] in ["response", "error"]
