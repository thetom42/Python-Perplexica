"""
Image Search Agent Module

This module provides functionality for performing image searches using multiple image search
engines (Bing Images, Google Images) through the SearxNG search engine interface.
"""

from typing import List, Dict, Any, Callable, Optional
from langchain.schema import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StrOutputParser
from backend.lib.searxng import search_searxng
from backend.utils.format_history import format_chat_history_as_string
from backend.utils.logger import logger

IMAGE_SEARCH_CHAIN_PROMPT = """
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

class RunnableSequence:
    """A simple implementation of a runnable sequence similar to TypeScript's RunnableSequence."""
    
    def __init__(self, steps: List[Callable]):
        """Initialize the RunnableSequence with a list of callable steps."""
        self.steps = steps
    
    async def invoke(self, input_data: Dict[str, Any]) -> Any:
        """Execute the sequence of steps on the input data."""
        result = input_data
        for step in self.steps:
            result = await step(result)
        return result

async def create_image_search_chain(llm: BaseChatModel) -> RunnableSequence:
    """
    Create the image search chain.
    
    Args:
        llm: The language model to use for query processing
        
    Returns:
        RunnableSequence: A chain that processes queries and retrieves image results
    """
    async def map_input(input_data: Dict[str, Any]) -> Dict[str, str]:
        """Map the input data to the format expected by the LLM chain."""
        return {
            "chat_history": format_chat_history_as_string(input_data["chat_history"]),
            "query": input_data["query"]
        }
    
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(IMAGE_SEARCH_CHAIN_PROMPT)
    )
    str_parser = StrOutputParser()
    
    async def run_chain(input_data: Dict[str, str]) -> str:
        """Run the LLM chain to process the query."""
        try:
            return await chain.arun(**input_data)
        except Exception as e:
            logger.error(f"Error in LLM chain: {str(e)}")
            return input_data["query"]  # Fallback to original query
    
    async def parse_output(output: str) -> str:
        """Parse the LLM output into a search query."""
        try:
            return str_parser.parse(output)
        except Exception as e:
            logger.error(f"Error parsing output: {str(e)}")
            return output  # Return unparsed output as fallback
    
    async def perform_search(query: str) -> List[Dict[str, str]]:
        """
        Perform the image search using multiple search engines.
        
        Args:
            query: The search query
            
        Returns:
            List[Dict[str, str]]: A list of image results, each containing img_src, url, and title
        """
        try:
            res = await search_searxng(query, {
                "engines": ["bing images", "google images"]
            })
            
            images = []
            for result in res["results"]:
                if result.get("img_src") and result.get("url") and result.get("title"):
                    images.append({
                        "img_src": result["img_src"],
                        "url": result["url"],
                        "title": result["title"]
                    })
            
            return images[:10]  # Return top 10 images
            
        except Exception as e:
            logger.error(f"Error in image search: {str(e)}")
            return []  # Return empty list on error
    
    return RunnableSequence([
        map_input,
        run_chain,
        parse_output,
        perform_search
    ])

async def handle_image_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    *args: Any,  # Accept but ignore additional arguments for compatibility
    **kwargs: Any
) -> List[Dict[str, str]]:
    """
    Handle an image search query.

    Args:
        query: The image search query
        history: The chat history
        llm: The language model to use for query processing
        *args: Additional positional arguments (ignored)
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        List[Dict[str, str]]: A list of image results, each containing:
            - img_src: URL of the image
            - url: URL of the page containing the image
            - title: Title of the image or containing page
            
    Note:
        This function accepts additional arguments for compatibility with other agents
        but only uses query, history, and llm.
    """
    try:
        image_search_chain = await create_image_search_chain(llm)
        return await image_search_chain.invoke({
            "chat_history": history,
            "query": query
        })
    except Exception as e:
        logger.error(f"Error in image search handler: {str(e)}")
        return []  # Return empty list on error
