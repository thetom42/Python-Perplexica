from agents.web_search_agent import handle_web_search
from agents.academic_search_agent import handle_academic_search
from agents.image_search_agent import handle_image_search
from agents.reddit_search_agent import handle_reddit_search
from agents.suggestion_generator_agent import handle_suggestion_generation
from agents.video_search_agent import handle_video_search
from agents.wolfram_alpha_search_agent import handle_wolfram_alpha_search
from agents.writing_assistant_agent import handle_writing_assistance
from agents.youtube_search_agent import handle_youtube_search

__all__ = [
    "handle_web_search",
    "handle_academic_search",
    "handle_image_search",
    "handle_reddit_search",
    "handle_suggestion_generation",
    "handle_video_search",
    "handle_wolfram_alpha_search",
    "handle_writing_assistance",
    "handle_youtube_search"
]
