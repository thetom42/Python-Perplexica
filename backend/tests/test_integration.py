import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from main import app
from .utils import create_base_request, create_test_chat_model, create_test_embedding_model

client = TestClient(app)


@pytest.mark.asyncio
async def test_search_to_suggestions_flow():
    """Test the flow from search to suggestions."""
    # First perform a search
    search_result = {
        "type": "response",
        "data": "Search result about AI"
    }
    
    with patch('routes.search.search_handlers.get', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = AsyncMock()
        mock_search.return_value.__aiter__.return_value = [search_result]
        
        search_request = {
            **create_base_request(
                create_test_chat_model(),
                create_test_embedding_model()
            ),
            "focus_mode": "web",
            "query": "Tell me about AI",
            "history": []
        }
        
        search_response = client.post("/api/search", json=search_request)
        assert search_response.status_code == 200

        # Then get suggestions based on the search
        with patch('routes.suggestions.handle_suggestion_generation', new_callable=AsyncMock) as mock_suggestions:
            mock_suggestions.return_value = ["Learn about machine learning", "Explore neural networks"]
            
            suggestion_request = {
                **create_base_request(),
                "chat_history": [
                    {"role": "user", "content": "Tell me about AI"},
                    {"role": "assistant", "content": "Search result about AI"}
                ]
            }
            
            suggestion_response = client.post("/api/suggestions", json=suggestion_request)
            assert suggestion_response.status_code == 200
            assert len(suggestion_response.json()["suggestions"]) == 2


@pytest.mark.asyncio
async def test_search_to_videos_flow():
    """Test the flow from search to related videos."""
    # First perform a search
    search_result = {
        "type": "response",
        "data": "Information about quantum computing"
    }
    
    with patch('routes.search.search_handlers.get', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = AsyncMock()
        mock_search.return_value.__aiter__.return_value = [search_result]
        
        search_request = {
            **create_base_request(),
            "focus_mode": "web",
            "query": "Explain quantum computing",
            "history": []
        }
        
        search_response = client.post("/api/search", json=search_request)
        assert search_response.status_code == 200

        # Then find related videos
        with patch('routes.videos.handle_video_search', new_callable=AsyncMock) as mock_videos:
            mock_videos.return_value = [
                {"title": "Quantum Computing Explained", "url": "http://example.com/video1"},
                {"title": "Introduction to Qubits", "url": "http://example.com/video2"}
            ]
            
            video_request = {
                **create_base_request(),
                "query": "quantum computing tutorial",
                "chat_history": [
                    {"role": "user", "content": "Explain quantum computing"},
                    {"role": "assistant", "content": "Information about quantum computing"}
                ]
            }
            
            video_response = client.post("/api/videos", json=video_request)
            assert video_response.status_code == 200
            assert len(video_response.json()["videos"]) == 2


@pytest.mark.asyncio
async def test_full_search_cycle():
    """Test a full search cycle including search, suggestions, and videos."""
    # Initial search
    search_result = {
        "type": "response",
        "data": "Information about space exploration"
    }
    
    with patch('routes.search.search_handlers.get', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = AsyncMock()
        mock_search.return_value.__aiter__.return_value = [search_result]
        
        search_request = {
            **create_base_request(),
            "focus_mode": "web",
            "query": "Tell me about space exploration",
            "history": []
        }
        
        search_response = client.post("/api/search", json=search_request)
        assert search_response.status_code == 200

        chat_history = [
            {"role": "user", "content": "Tell me about space exploration"},
            {"role": "assistant", "content": "Information about space exploration"}
        ]

        # Get suggestions
        with patch('routes.suggestions.handle_suggestion_generation', new_callable=AsyncMock) as mock_suggestions:
            mock_suggestions.return_value = [
                "Learn about Mars missions",
                "Explore SpaceX launches"
            ]
            
            suggestion_request = {
                **create_base_request(),
                "chat_history": chat_history
            }
            
            suggestion_response = client.post("/api/suggestions", json=suggestion_request)
            assert suggestion_response.status_code == 200
            suggestions = suggestion_response.json()["suggestions"]
            assert len(suggestions) == 2

            # Find related videos based on a suggestion
            with patch('routes.videos.handle_video_search', new_callable=AsyncMock) as mock_videos:
                mock_videos.return_value = [
                    {"title": "Mars Mission Overview", "url": "http://example.com/video1"},
                    {"title": "SpaceX Launch Highlights", "url": "http://example.com/video2"}
                ]
                
                video_request = {
                    **create_base_request(),
                    "query": suggestions[0],  # Use first suggestion as video query
                    "chat_history": chat_history
                }
                
                video_response = client.post("/api/videos", json=video_request)
                assert video_response.status_code == 200
                assert len(video_response.json()["videos"]) == 2
