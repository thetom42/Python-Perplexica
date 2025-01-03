"""
Video Search Agent Module

This module provides functionality for performing video searches using the SearxNG search engine
with YouTube as the primary source. It specifically handles video content with support for
thumbnails and embedded iframes.
"""

from typing import List, Dict, Any, Literal
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from lib.searxng import search_searxng
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

VIDEO_SEARCH_CHAIN_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question so it is a standalone question that can be used by the LLM to search Youtube for videos.
You need to make sure the rephrased question agrees with the conversation and is relevant to the conversation.

Example:
1. Follow up question: How does a car work?
Rephrased: How does a car work?

2. Follow up question: What is the theory of relativity?
Rephrased: What is theory of relativity

3. Follow up question: How does an AC work?
Rephrased: How does an AC work

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

class VideoSearchAgent(AbstractAgent):
    """Agent for performing video searches and retrieving results."""

    async def perform_search(self, query: str) -> Dict[str, Any]:
        """Perform the video search using YouTube through SearxNG."""
        try:
            res = await search_searxng(query, {
                "engines": ["youtube"]
            })

            documents = []
            for result in res["results"]:
                if (result.get("thumbnail") and
                    result.get("url") and
                    result.get("title") and
                    result.get("iframe_src")):
                    documents.append(Document(
                        page_content="",  # Video searches don't need text content
                        metadata={
                            "img_src": result["thumbnail"],
                            "url": result["url"],
                            "title": result["title"],
                            "iframe_src": result["iframe_src"]
                        }
                    ))

            return {"query": query, "docs": documents[:10]}  # Return top 10 videos

        except Exception as e:
            logger.error("Error in video search: %s", str(e))
            return {"query": query, "docs": []}

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the video search retriever chain."""
        prompt = PromptTemplate.from_template(VIDEO_SEARCH_CHAIN_PROMPT)
        str_parser = StrOutputParser()
        chain = prompt | self.llm | str_parser

        retriever_chain = RunnableSequence(
            RunnableParallel({
                "rephrased_query": chain,
                "original_input": RunnablePassthrough()
            }),
            lambda x: self.perform_search(x["rephrased_query"])
        )

        return retriever_chain

    async def process_results(self, input_data: Dict[str, Any], retriever_chain: RunnableSequence) -> Dict[str, Any]:
        """Process the retriever results into the final format."""
        try:
            try:
                result = await retriever_chain.invoke(input_data)
            except Exception as e:
                logger.error("Error creating retriever chain: %s", str(e))
                raise

            if not result["docs"]:
                return {
                    "type": "response",
                    "data": []
                }

            # Extract video data from documents
            videos = [
                {
                    "img_src": doc.metadata["img_src"],
                    "url": doc.metadata["url"],
                    "title": doc.metadata["title"],
                    "iframe_src": doc.metadata["iframe_src"]
                }
                for doc in result["docs"]
            ]

            return {
                "type": "response",
                "data": videos
            }

        except Exception as e:
            logger.error("Error creating answering chain: %s", str(e))
            return {
                "type": "error",
                "data": "An error occurred while processing the video search results"
            }

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the video search answering chain."""
        retriever_chain = await self.create_retriever_chain()
        return RunnableSequence(
            RunnablePassthrough(),
            lambda x: self.process_results(x, retriever_chain)
        )

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        if not query.strip() or not context.strip():
            raise ValueError("Query and context cannot be empty")
            
        base_prompt = self.create_base_prompt()
        return f"""
        {base_prompt}
        
        You are set on focus mode 'Video Search', this means you will be searching for relevant videos on YouTube.
        The context provided contains video search results from YouTube.
        
        User Query: {query}
        Context: {context}
        """

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        if response.lower().startswith("error:"):
            return {
                "type": "error",
                "data": response[6:].strip()  # Remove "error:" prefix
            }
        return {
            "type": "response",
            "data": response
        }

async def handle_video_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> List[Dict[str, str]]:
    """
    Handle a video search query.

    Args:
        query: The video search query
        history: The chat history
        llm: The language model to use for query processing
        embeddings: The embeddings model (unused for video search)
        optimization_mode: The optimization mode (unused for video search)

    Returns:
        List[Dict[str, str]]: A list of video results, each containing:
            - img_src: URL of the video thumbnail
            - url: URL of the video page
            - title: Title of the video
            - iframe_src: Embedded iframe URL for the video

    Raises:
        TypeError: If any input parameters are of invalid type
        ValueError: If query is empty
    """
    if not isinstance(query, str) or not isinstance(history, list):
        raise TypeError("Invalid input types")
    
    if not query.strip():
        raise ValueError("Query cannot be empty")

    agent = VideoSearchAgent(llm, embeddings, optimization_mode)
    results = []
    
    try:
        try:
            async for result in agent.handle_search(query, history):
                if result["type"] == "response":
                    results.extend(result["data"])
                elif result["type"] == "error":
                    logger.error("Error in video search: %s", result["data"])
                    break
        except Exception as e:
            logger.error("Error in video search: %s", str(e))
            raise
    except Exception as e:
        logger.error("Error in video search: %s", str(e))
        
    return results
