"""
Image Search Agent Module

This module provides functionality for performing image searches using the SearxNG search engine
and processing the results using a language model to generate descriptions of the search results.
"""

from langchain.schema import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from backend.lib.searxng import search_searxng
from typing import List, Dict, Any

async def handle_image_search(query: str, history: List[BaseMessage], llm: BaseChatModel, embeddings: Embeddings) -> Dict[str, Any]:
    """
    Handle an image search query and generate a response based on search results.

    This function performs the following steps:
    1. Executes an image search query using SearxNG with Google Images as the engine.
    2. Processes the search results to create a list of image information.
    3. Uses a language model to generate a description of the image search results.

    Args:
        query (str): The image search query.
        history (List[BaseMessage]): The chat history (not used in the current implementation).
        llm (BaseChatModel): The language model to use for generating the response.
        embeddings (Embeddings): The embeddings model (not used in the current implementation).

    Returns:
        Dict[str, Any]: A dictionary containing the type of response, the generated description,
                        and a list of image results.
    """
    # Implement image search logic here
    search_results = await search_searxng(query, {"engines": "google_images"})

    image_results = [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "thumbnail": result.get("thumbnail", "")
        }
        for result in search_results.get("results", [])
    ]

    response_generator = LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_messages([
            ("system", """
            You are an AI image search assistant. Describe the image search results based on the provided information.
            Focus on the content, style, and relevance of the images to the query.
            If no relevant images are found, suggest refining the search query.
            """),
            ("human", "{query}"),
            ("human", "Image search results:\n{context}")
        ])
    )

    context = "\n".join([f"{i+1}. {result['title']} - {result['url']}" for i, result in enumerate(image_results)])
    response = await response_generator.arun(query=query, context=context)

    return {
        "type": "response",
        "data": response,
        "images": image_results
    }

# Additional helper functions can be implemented as needed
