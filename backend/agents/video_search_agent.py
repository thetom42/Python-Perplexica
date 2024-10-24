"""
Video Search Agent Module

This module provides functionality for performing video searches using the SearxNG search engine
with YouTube as the primary source. It specifically handles video content with support for
thumbnails and embedded iframes.
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

async def create_video_search_chain(llm: BaseChatModel) -> RunnableSequence:
    """
    Create the video search chain.
    
    Args:
        llm: The language model to use for query processing
        
    Returns:
        RunnableSequence: A chain that processes queries and retrieves video results
    """
    async def map_input(input_data: Dict[str, Any]) -> Dict[str, str]:
        """Map the input data to the format expected by the LLM chain."""
        return {
            "chat_history": format_chat_history_as_string(input_data["chat_history"]),
            "query": input_data["query"]
        }
    
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(VIDEO_SEARCH_CHAIN_PROMPT)
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
        Perform the video search using YouTube through SearxNG.
        
        Args:
            query: The search query
            
        Returns:
            List[Dict[str, str]]: A list of video results, each containing:
                - img_src: URL of the video thumbnail
                - url: URL of the video page
                - title: Title of the video
                - iframe_src: Embedded iframe URL for the video
        """
        try:
            res = await search_searxng(query, {
                "engines": ["youtube"]
            })
            
            videos = []
            for result in res["results"]:
                if (result.get("thumbnail") and 
                    result.get("url") and 
                    result.get("title") and 
                    result.get("iframe_src")):
                    videos.append({
                        "img_src": result["thumbnail"],
                        "url": result["url"],
                        "title": result["title"],
                        "iframe_src": result["iframe_src"]
                    })
            
            return videos[:10]  # Return top 10 videos
            
        except Exception as e:
            logger.error(f"Error in video search: {str(e)}")
            return []  # Return empty list on error
    
    return RunnableSequence([
        map_input,
        run_chain,
        parse_output,
        perform_search
    ])

async def handle_video_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    *args: Any,  # Accept but ignore additional arguments for compatibility
    **kwargs: Any
) -> List[Dict[str, str]]:
    """
    Handle a video search query.

    Args:
        query: The video search query
        history: The chat history
        llm: The language model to use for query processing
        *args: Additional positional arguments (ignored)
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        List[Dict[str, str]]: A list of video results, each containing:
            - img_src: URL of the video thumbnail
            - url: URL of the video page
            - title: Title of the video
            - iframe_src: Embedded iframe URL for the video
            
    Note:
        This function accepts additional arguments for compatibility with other agents
        but only uses query, history, and llm.
    """
    try:
        video_search_chain = await create_video_search_chain(llm)
        return await video_search_chain.invoke({
            "chat_history": history,
            "query": query
        })
    except Exception as e:
        logger.error(f"Error in video search handler: {str(e)}")
        return []  # Return empty list on error
