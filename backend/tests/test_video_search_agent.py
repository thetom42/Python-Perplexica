import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain.schema import HumanMessage, AIMessage
from agents.video_search_agent import (
    handle_video_search,
    VideoSearchAgent
)

@pytest.fixture
def mock_chat_model():
    mock = AsyncMock()
    mock.arun = AsyncMock()
    return mock

@pytest.fixture
def mock_embeddings_model():
    return AsyncMock()

@pytest.fixture
def video_agent(mock_chat_model, mock_embeddings_model):
    return VideoSearchAgent(mock_chat_model, mock_embeddings_model, "balanced")

@pytest.fixture
def sample_video_results():
    return {
        "results": [
            {
                "title": "Python Tutorial",
                "url": "https://youtube.com/watch?v=1",
                "thumbnail": "https://i.ytimg.com/vi/1/default.jpg",
                "iframe_src": "https://www.youtube.com/embed/1"
            },
            {
                "title": "Advanced Python",
                "url": "https://youtube.com/watch?v=2",
                "thumbnail": "https://i.ytimg.com/vi/2/default.jpg",
                "iframe_src": "https://www.youtube.com/embed/2"
            }
        ]
    }

@pytest.mark.asyncio
async def test_handle_video_search_basic(mock_chat_model, mock_embeddings_model):
    query = "Python programming tutorials"
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!")
    ]

    with patch('backend.agents.video_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {
            "results": [
                {
                    "title": "Python Tutorial",
                    "url": "https://youtube.com/watch?v=1",
                    "thumbnail": "https://i.ytimg.com/vi/1/default.jpg",
                    "iframe_src": "https://www.youtube.com/embed/1"
                }
            ]
        }

        result = await handle_video_search(query, history, mock_chat_model, mock_embeddings_model)

        assert len(result) > 0
        assert "title" in result[0]
        assert "url" in result[0]
        assert "img_src" in result[0]
        assert "iframe_src" in result[0]

@pytest.mark.asyncio
async def test_retriever_chain_creation(video_agent):
    chain = await video_agent.create_retriever_chain()
    assert chain is not None

    with patch('backend.agents.video_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}

        result = await chain.invoke({
            "query": "test query",
            "chat_history": []
        })

        assert "query" in result
        assert "docs" in result

@pytest.mark.asyncio
async def test_query_rephrasing(mock_chat_model, mock_embeddings_model):
    query = "Show me how to make a cake"
    history = [
        HumanMessage(content="I want to learn cooking"),
        AIMessage(content="I can help with that!")
    ]

    with patch('backend.agents.video_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}
        mock_chat_model.arun.return_value = "cake making tutorial"

        await handle_video_search(query, history, mock_chat_model, mock_embeddings_model)

        # Verify the query was processed through the rephrasing chain
        assert mock_chat_model.arun.called
        call_args = mock_chat_model.arun.call_args[1]
        assert "chat_history" in call_args
        assert "query" in call_args

@pytest.mark.asyncio
async def test_youtube_specific_search(mock_chat_model, mock_embeddings_model):
    query = "test query"
    history = []

    with patch('backend.agents.video_search_agent.search_searxng') as mock_search:
        mock_search.return_value = {"results": []}

        await handle_video_search(query, history, mock_chat_model, mock_embeddings_model)

        # Verify YouTube-specific search configuration
        mock_search.assert_called_once()
        call_args = mock_search.call_args[1]
        assert call_args.get("engines") == ["youtube"]

@pytest.mark.asyncio
async def test_error_handling(mock_chat_model, mock_embeddings_model):
    query = "test query"
    history = []

    with patch('backend.agents.video_search_agent.search_searxng') as mock_search:
        mock_search.side_effect = Exception("Search API error")

        result = await handle_video_search(query, history, mock_chat_model, mock_embeddings_model)
        assert len(result) == 0  # Should return empty list on error

@pytest.mark.asyncio
async def test_empty_query_handling(mock_chat_model, mock_embeddings_model):
    query = ""
    history = []

    result = await handle_video_search(query, history, mock_chat_model, mock_embeddings_model)
    assert isinstance(result, list)
    assert len(result) == 0

@pytest.mark.asyncio
async def test_video_result_fields(mock_chat_model, mock_embeddings_model, sample_video_results):
    query = "test query"
    history = []

    with patch('backend.agents.video_search_agent.search_searxng') as mock_search:
        mock_search.return_value = sample_video_results

        result = await handle_video_search(query, history, mock_chat_model, mock_embeddings_model)

        assert len(result) == len(sample_video_results["results"])
        for video in result:
            assert "title" in video
            assert "url" in video
            assert "img_src" in video
            assert "iframe_src" in video
            assert video["url"].startswith("https://youtube.com/watch?v=")
            assert video["iframe_src"].startswith("https://www.youtube.com/embed/")

@pytest.mark.asyncio
async def test_result_limit(mock_chat_model, mock_embeddings_model):
    query = "test query"
    history = []

    with patch('backend.agents.video_search_agent.search_searxng') as mock_search:
        # Create more than 10 results
        results = {
            "results": [
                {
                    "title": f"Video {i}",
                    "url": f"https://youtube.com/watch?v={i}",
                    "thumbnail": f"https://i.ytimg.com/vi/{i}/default.jpg",
                    "iframe_src": f"https://www.youtube.com/embed/{i}"
                }
                for i in range(15)
            ]
        }
        mock_search.return_value = results

        result = await handle_video_search(query, history, mock_chat_model, mock_embeddings_model)
        assert len(result) <= 10  # Should be limited to 10 results

@pytest.mark.asyncio
async def test_malformed_search_results(mock_chat_model, mock_embeddings_model):
    query = "test query"
    history = []

    with patch('backend.agents.video_search_agent.search_searxng') as mock_search:
        # Missing or incomplete fields in results
        mock_search.return_value = {
            "results": [
                {"title": "Test"},  # Missing required fields
                {"url": "https://youtube.com"},  # Missing required fields
                {
                    "title": "Complete Video",
                    "url": "https://youtube.com/watch?v=1",
                    "thumbnail": "https://i.ytimg.com/vi/1/default.jpg",
                    "iframe_src": "https://www.youtube.com/embed/1"
                }
            ]
        }

        result = await handle_video_search(query, history, mock_chat_model, mock_embeddings_model)

        # Should only include complete results
        assert len(result) == 1
        assert result[0]["title"] == "Complete Video"

@pytest.mark.asyncio
async def test_video_agent_methods(video_agent):
    # Test format prompt
    prompt = video_agent.format_prompt("test", "context")
    assert prompt == ""  # Video agent doesn't use text prompts

    # Test parse response
    parsed = await video_agent.parse_response("test response")
    assert parsed["type"] == "response"
    assert parsed["data"] == "test response"
