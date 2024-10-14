"""
YouTube Search Agent Module

This module provides functionality for performing YouTube searches using the SearxNG search engine
and processing the results using a language model to generate summaries of the video search results.
"""

from langchain.schema import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from backend.lib.searxng import search_searxng
from typing import List, Dict, Any

async def handle_youtube_search(query: str, history: List[BaseMessage], llm: BaseChatModel, embeddings: Embeddings) -> Dict[str, Any]:
    """
    Handle a YouTube search query and generate a response based on search results.

    This function performs the following steps:
    1. Executes a YouTube search query using SearxNG with YouTube as the engine.
    2. Processes the search results to create a list of YouTube video information.
    3. Uses a language model to generate a summary of the YouTube search results.

    Args:
        query (str): The YouTube search query.
        history (List[BaseMessage]): The chat history (not used in the current implementation).
        llm (BaseChatModel): The language model to use for generating the response.
        embeddings (Embeddings): The embeddings model (not used in the current implementation).

    Returns:
        Dict[str, Any]: A dictionary containing the type of response, the generated summary,
                        and a list of YouTube video results.
    """
    # Implement YouTube search logic here
    search_results = await search_searxng(query, {"engines": "youtube"})

    youtube_results = [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "thumbnail": result.get("thumbnail", ""),
            "description": result.get("content", ""),
            "channel": result.get("author", ""),
            "published_date": result.get("publishedDate", "")
        }
        for result in search_results.get("results", [])
    ]

    response_generator = LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_messages([
            ("system", """
            You are an AI YouTube search assistant. Summarize the YouTube search results based on the provided information.
            Focus on the main topics, content creators, video quality, and relevance to the query.
            Highlight any popular or highly-rated videos, and mention any recurring themes or topics across multiple videos.
            If no relevant videos are found, suggest refining the search query or exploring related channels or topics.
            Use an informative and engaging tone, and provide a brief overview of what the user can expect from these videos.
            """),
            ("human", "{query}"),
            ("human", "YouTube search results:\n{context}")
        ])
    )

    context = "\n".join([f"{i+1}. {result['title']} by {result['channel']} - {result['description'][:150]}..." for i, result in enumerate(youtube_results)])
    response = await response_generator.arun(query=query, context=context)

    return {
        "type": "response",
        "data": response,
        "youtube_videos": youtube_results
    }

# Additional helper functions can be implemented as needed
