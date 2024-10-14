"""
Video Search Agent Module

This module provides functionality for performing video searches using the SearxNG search engine
with YouTube as the primary source. It processes the results using a language model to generate
summaries of the video search results.
"""

from langchain.schema import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from backend.lib.searxng import search_searxng
from typing import List, Dict, Any

async def handle_video_search(query: str, history: List[BaseMessage], llm: BaseChatModel, embeddings: Embeddings) -> Dict[str, Any]:
    """
    Handle a video search query and generate a response based on search results.

    This function performs the following steps:
    1. Executes a video search query using SearxNG with YouTube as the engine.
    2. Processes the search results to create a list of video information.
    3. Uses a language model to generate a summary of the video search results.

    Args:
        query (str): The video search query.
        history (List[BaseMessage]): The chat history (not used in the current implementation).
        llm (BaseChatModel): The language model to use for generating the response.
        embeddings (Embeddings): The embeddings model (not used in the current implementation).

    Returns:
        Dict[str, Any]: A dictionary containing the type of response, the generated summary,
                        and a list of video results.
    """
    # Implement video search logic here
    search_results = await search_searxng(query, {"engines": "youtube"})

    video_results = [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "thumbnail": result.get("thumbnail", ""),
            "description": result.get("content", "")
        }
        for result in search_results.get("results", [])
    ]

    response_generator = LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_messages([
            ("system", """
            You are an AI video search assistant. Summarize the video search results based on the provided information.
            Focus on the main topics, content creators, and relevance of the videos to the query.
            If no relevant videos are found, suggest refining the search query or exploring related topics.
            Use an informative and engaging tone, highlighting key points from video descriptions.
            """),
            ("human", "{query}"),
            ("human", "Video search results:\n{context}")
        ])
    )

    context = "\n".join([f"{i+1}. {result['title']} - {result['description'][:200]}..." for i, result in enumerate(video_results)])
    response = await response_generator.arun(query=query, context=context)

    return {
        "type": "response",
        "data": response,
        "videos": video_results
    }

# Additional helper functions can be implemented as needed
