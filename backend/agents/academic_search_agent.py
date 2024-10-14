"""
Academic Search Agent Module

This module provides functionality for performing academic searches using the SearxNG search engine
and processing the results using a language model to generate concise summaries.
"""

from langchain.schema import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from backend.lib.searxng import search_searxng
from backend.utils.compute_similarity import compute_similarity
from backend.utils.format_history import format_chat_history_as_string
from typing import List, Dict, Any

async def handle_academic_search(query: str, history: List[BaseMessage], llm: BaseChatModel, embeddings: Embeddings) -> Dict[str, Any]:
    """
    Handle an academic search query and generate a response based on search results.

    This function performs the following steps:
    1. Executes a search query using SearxNG with Google Scholar as the engine.
    2. Processes the search results to create a context.
    3. Uses a language model to generate a concise summary of the academic search results.

    Args:
        query (str): The academic search query.
        history (List[BaseMessage]): The chat history (not used in the current implementation).
        llm (BaseChatModel): The language model to use for generating the response.
        embeddings (Embeddings): The embeddings model (not used in the current implementation).

    Returns:
        Dict[str, Any]: A dictionary containing the type of response and the generated summary.
    """
    # Implement academic search logic here
    search_results = await search_searxng(query, {"engines": "google_scholar"})

    context = "\n".join([f"{i+1}. {result['content']}" for i, result in enumerate(search_results['results'])])

    response_generator = LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_messages([
            ("system", """
            You are an AI research assistant specializing in academic literature.
            Provide a concise summary of the academic search results, highlighting key findings and methodologies.
            Use an objective and scholarly tone. Cite sources using [number] notation.
            If there's insufficient information, state that more research may be needed.
            """),
            ("human", "{query}"),
            ("human", "Academic search results:\n{context}")
        ])
    )

    response = await response_generator.arun(query=query, context=context)

    return {"type": "response", "data": response}

# Additional helper functions can be implemented as needed
