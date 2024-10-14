from .web_search_agent import handle_web_search
from .academic_search_agent import handle_academic_search
from .image_search_agent import handle_image_search
from .reddit_search_agent import handle_reddit_search
from .suggestion_generator_agent import handle_suggestion_generation
from .video_search_agent import handle_video_search
from .wolfram_alpha_search_agent import handle_wolfram_alpha_search
from .writing_assistant_agent import handle_writing_assistance
from .youtube_search_agent import handle_youtube_search

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
