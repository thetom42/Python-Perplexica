"""
Web Search Agent Module

This module provides functionality for performing web searches, rephrasing queries,
and generating responses based on search results or webpage content. It uses the SearxNG
search engine and a language model to process and summarize information.
"""

from langchain.schema import BaseMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import LLMChain
from langchain.embeddings.base import Embeddings
from langchain.chat_models.base import BaseChatModel
from backend.lib.searxng import search_searxng
from backend.utils.compute_similarity import compute_similarity
from backend.utils.format_history import format_chat_history_as_string
from backend.lib.link_document import get_documents_from_links
from backend.utils.logger import logger
from backend.utils.cache import cache_result
from typing import List, Dict, Any

@cache_result(expire_time=3600)  # Cache results for 1 hour
async def handle_web_search(query: str, history: List[BaseMessage], llm: BaseChatModel, embeddings: Embeddings) -> Dict[str, Any]:
    """
    Handle a web search query, rephrase it if necessary, perform the search or fetch webpage content,
    and generate a response based on the results.

    This function performs the following steps:
    1. Rephrase the query to make it standalone or identify if it's a URL summarization request.
    2. Perform a web search using SearxNG or fetch webpage content if a URL is provided.
    3. Generate a response using a language model based on the search results or webpage content.

    Args:
        query (str): The user's search query.
        history (List[BaseMessage]): The chat history.
        llm (BaseChatModel): The language model to use for generating responses.
        embeddings (Embeddings): The embeddings model (not used in the current implementation).

    Returns:
        Dict[str, Any]: A dictionary containing the type of response and the generated answer or error message.
    """
    try:
        logger.info("Starting web search for query: %s", query)

        # Step 1: Rephrase the query
        rephraser = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template="""
                You are an AI question rephraser. Rephrase the follow-up question so it is a standalone question.
                If it's a simple writing task or greeting, return "not_needed".
                If the user asks about a specific URL or wants to summarize a webpage, return the URL and "summarize" as the question.

                Chat history:
                {chat_history}

                Follow up question: {query}
                Rephrased question:
                """,
                input_variables=["chat_history", "query"]
            )
        )

        rephrased_query = await rephraser.arun(chat_history=format_chat_history_as_string(history), query=query)
        logger.debug("Rephrased query: %s", rephrased_query)

        if rephrased_query.lower() == "not_needed":
            logger.info("Query doesn't require web search")
            return {"type": "response", "data": "I don't need to search for that. How can I assist you?"}

        # Step 2: Perform web search or fetch document content
        if "http" in rephrased_query.lower():
            logger.info("Fetching content from URL: %s", rephrased_query.split()[-1])
            url = rephrased_query.split()[-1]
            docs = await get_documents_from_links([url])
            context = "\n".join([doc.page_content for doc in docs])
        else:
            logger.info("Performing web search for: %s", rephrased_query)
            search_results = await search_searxng(rephrased_query)
            context = "\n".join([f"{i+1}. {result['content']}" for i, result in enumerate(search_results['results'])])

        # Step 3: Generate response
        response_generator = LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_messages([
                ("system", """
                You are Perplexica, an AI model expert at searching the web and answering queries.
                Generate an informative and relevant response based on the provided context.
                Use an unbiased and journalistic tone. Do not repeat the text.
                Cite your sources using [number] notation at the end of each sentence.
                If there's nothing relevant in the search results, say "I couldn't find any relevant information on this topic."
                """),
                ("human", "{query}"),
                ("human", "Context:\n{context}")
            ])
        )

        response = await response_generator.arun(query=query, context=context)
        logger.info("Generated response for web search query")

        return {"type": "response", "data": response}

    except Exception as e:
        logger.error("Error in handle_web_search: %s", str(e), exc_info=True)
        return {"type": "error", "data": "An error occurred while processing your request. Please try again later."}

# You may need to implement additional helper functions and classes as needed
