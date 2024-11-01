"""
Image Search Agent Module

This module provides functionality for performing image searches, with query rephrasing
and result processing to ensure relevant and high-quality image results.
"""

from typing import Dict, Any, List, AsyncGenerator
from langchain.schema import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
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

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the image search retriever chain."""
        prompt = PromptTemplate.from_template(IMAGE_SEARCH_PROMPT)

        # Set temperature to 0 for more precise rephrasing
        if hasattr(self.llm, "temperature"):
            self.llm.temperature = 0

        chain = prompt | self.llm

        async def process_output(llm_output: Any) -> Dict[str, Any]:
            """Process the LLM output and perform the image search."""
            try:
                rephrased_query = llm_output.content if hasattr(llm_output, 'content') else str(llm_output)

                results = await search_searxng(rephrased_query, {
                    "engines": ["bing images", "google images"]
                })

                if not results or not results.get("results"):
                    return {"query": rephrased_query, "images": []}

                image_results = []
                for result in results["results"]:
                    if all(key in result for key in ["img_src", "url", "title"]):
                        image_results.append({
                            "img_src": result["img_src"],
                            "url": result["url"],
                            "title": result["title"]
                        })

                # Limit to 10 results
                image_results = image_results[:10]

                return {"query": rephrased_query, "images": image_results}
            except Exception as e:
                logger.error("Error in image search/processing: %s", str(e))
                return {"query": "", "images": []}

        retriever_chain = RunnableSequence([
            {
                "llm_output": chain,
                "original_input": RunnablePassthrough()
            },
            lambda x: process_output(x["llm_output"])
        ])

        return retriever_chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the image search answering chain."""
        retriever_chain = await self.create_retriever_chain()

        async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate the final response using the retriever chain."""
            try:
                retriever_result = await retriever_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                if not retriever_result["query"]:
                    return {
                        "type": "error",
                        "data": "An error occurred during image search"
                    }

                if not retriever_result["images"]:
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

        return RunnableSequence([generate_response])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        # Not used for image search but required by AbstractAgent
        return ""

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        # Not used for image search but required by AbstractAgent
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
