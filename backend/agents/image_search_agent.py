"""
Image Search Agent Module

This module provides functionality for performing image searches, with query rephrasing
and result processing to ensure relevant and high-quality image results.
"""

from typing import Dict, Any, List, AsyncGenerator
from langchain.schema import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from lib.searxng import search_searxng
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

IMAGE_SEARCH_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question so it is a standalone question that can be used by the LLM to search the web for images.
You need to make sure the rephrased question agrees with the conversation and is relevant to the conversation.

Example:
1. Follow up question: What is a cat?
Rephrased: A cat

2. Follow up question: What is a car? How does it works?
Rephrased: Car working

3. Follow up question: How does an AC work?
Rephrased: AC working

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

class ImageSearchAgent(AbstractAgent):
    """Agent for performing image searches and processing results."""

    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data types."""
        query = input_data.get("query")
        chat_history = input_data.get("chat_history")

        if query is None or not isinstance(query, str):
            raise TypeError("Query must be a string")
        if not isinstance(chat_history, list):
            raise TypeError("Chat history must be a list")

        return input_data

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the image search retriever chain."""
        if not hasattr(self, 'llm') or not isinstance(self.llm, BaseChatModel):
            raise TypeError("llm must be an instance of BaseChatModel")
            
        prompt = PromptTemplate.from_template(IMAGE_SEARCH_PROMPT)

        # Set temperature to 0 for more precise rephrasing
        if hasattr(self.llm, "temperature"):
            self.llm.temperature = 0

        async def process_query(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process the query and perform the search."""
            try:
                query = input_data["query"]
                # Handle empty query
                if not query.strip():
                    return {"query": "", "images": [], "response": None}

                # Rephrase query
                formatted_prompt = prompt.format(
                    query=query,
                    chat_history=input_data.get("chat_history", [])
                )
                llm_output = await self.llm.ainvoke(formatted_prompt)
                rephrased_query = llm_output.content if hasattr(llm_output, 'content') else str(llm_output)
                rephrased_query = rephrased_query.strip()

                try:
                    # Perform search
                    results = await search_searxng(rephrased_query, {
                        "engines": ["bing images", "google images"]
                    })

                    if not results or not results.get("results"):
                        return {"query": rephrased_query, "images": [], "response": None}

                    image_results = []
                    for result in results["results"]:
                        if all(key in result for key in ["img_src", "url", "title"]):
                            image_results.append({
                                "img_src": result["img_src"],
                                "url": result["url"],
                                "title": result["title"]
                            })

                    # Limit to 10 results
                    return {
                        "query": rephrased_query,
                        "images": image_results[:10],
                        "response": None
                    }

                except Exception as e:
                    error_msg = str(e)
                    logger.error("Error in image search: %s", error_msg)
                    if "rate limit" in error_msg.lower():
                        return {
                            "query": "",
                            "images": [],
                            "response": f"Rate limit error: {error_msg}"
                        }
                    return {"query": "", "images": [], "response": None}

            except Exception as e:
                logger.error("Error in query processing: %s", str(e))
                return {"query": "", "images": [], "response": None}

        # Create the chain with validation first
        chain = RunnablePassthrough() | self.validate_input | process_query
        return chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the image search answering chain."""
        retriever_chain = await self.create_retriever_chain()

        async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate the final response using the retriever chain."""
            try:
                retriever_result = await retriever_chain.ainvoke(input_data)

                # Handle rate limit errors or other errors with response
                if retriever_result.get("response"):
                    return {
                        "type": "error",
                        "data": retriever_result["response"]
                    }

                # Handle empty results
                if not retriever_result.get("images"):
                    return {
                        "type": "error",
                        "data": "No images found. Please try a different search query."
                    }

                return {
                    "type": "images",
                    "data": retriever_result["images"]
                }
            except Exception as e:
                logger.error("Error in response generation: %s", str(e))
                return {
                    "type": "error",
                    "data": "An error occurred while processing the image search results"
                }

        # Create the chain with validation first
        chain = RunnablePassthrough() | self.validate_input | generate_response
        return chain

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        base_prompt = self.create_base_prompt()
        return f"""
        {base_prompt}
        
        You are set on focus mode 'Image Search', this means you will be searching for relevant images based on the user's query.
        """

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response
        }

async def handle_image_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: str = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle an image search query and generate streaming responses.

    Args:
        query: The image search query
        history: The chat history
        llm: The language model to use
        embeddings: The embeddings model (not used for image search but required for consistency)
        optimization_mode: The mode determining the trade-off between speed and quality

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = ImageSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result
