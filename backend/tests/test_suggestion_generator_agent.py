import pytest
from unittest.mock import AsyncMock, patch
from langchain.schema import HumanMessage, AIMessage
from backend.agents.suggestion_generator_agent import generate_suggestions, SuggestionGeneratorAgent

@pytest.fixture
def mock_chat_model():
    return AsyncMock()

@pytest.fixture
def mock_embeddings_model():
    return AsyncMock()

@pytest.fixture
def suggestion_agent(mock_chat_model, mock_embeddings_model):
    return SuggestionGeneratorAgent(mock_chat_model, mock_embeddings_model, "balanced")

@pytest.mark.asyncio
async def test_generate_suggestions(mock_chat_model, mock_embeddings_model):
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I help you with suggestions today?")
    ]

    mock_chat_model.arun.return_value = """<suggestions>
Tell me about healthy breakfast options
What are some quick breakfast recipes?
How can I make a balanced breakfast?
What should I avoid for breakfast?
</suggestions>"""

    suggestions = await generate_suggestions(history, mock_chat_model, mock_embeddings_model)

    assert isinstance(suggestions, list)
    assert len(suggestions) > 0
    assert any("breakfast" in suggestion.lower() for suggestion in suggestions)
    mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_generate_suggestions_empty_history(mock_chat_model, mock_embeddings_model):
    history = []

    mock_chat_model.arun.return_value = """<suggestions>
How can I help you today?
What topics interest you?
Would you like to learn something new?
</suggestions>"""

    suggestions = await generate_suggestions(history, mock_chat_model, mock_embeddings_model)

    assert isinstance(suggestions, list)
    assert len(suggestions) > 0
    mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_generate_suggestions_with_context(mock_chat_model, mock_embeddings_model):
    history = [
        HumanMessage(content="I'm planning a corporate retreat"),
        AIMessage(content="That sounds exciting! How can I assist you with planning your corporate retreat?")
    ]

    mock_chat_model.arun.return_value = """<suggestions>
What team-building activities would work best?
How long should the retreat be?
What's a good location for a corporate retreat?
What's the ideal group size for team activities?
</suggestions>"""

    suggestions = await generate_suggestions(history, mock_chat_model, mock_embeddings_model)

    assert isinstance(suggestions, list)
    assert len(suggestions) > 0
    assert any("team" in suggestion.lower() for suggestion in suggestions)
    mock_chat_model.arun.assert_called_once()

@pytest.mark.asyncio
async def test_generate_suggestions_error_handling(mock_chat_model, mock_embeddings_model):
    history = [HumanMessage(content="Hello")]
    mock_chat_model.arun.side_effect = Exception("Model error")

    suggestions = await generate_suggestions(history, mock_chat_model, mock_embeddings_model)

    assert isinstance(suggestions, list)
    assert len(suggestions) == 1
    assert "unable to generate suggestions" in suggestions[0].lower()

@pytest.mark.asyncio
async def test_suggestion_generator_agent_methods(suggestion_agent):
    # Test retriever chain
    retriever_chain = await suggestion_agent.create_retriever_chain()
    result = await retriever_chain.invoke({"query": "test", "chat_history": []})
    assert "query" in result
    assert "docs" in result
    assert result["docs"] == []  # Should return empty docs list

    # Test answering chain
    answering_chain = await suggestion_agent.create_answering_chain()
    assert answering_chain is not None

    # Test format prompt
    prompt = suggestion_agent.format_prompt("test", "context")
    assert "You are an AI suggestion generator" in prompt

    # Test parse response
    parsed = await suggestion_agent.parse_response("suggestion1\nsuggestion2")
    assert parsed["type"] == "response"
    assert isinstance(parsed["data"], list)
    assert len(parsed["data"]) == 2
