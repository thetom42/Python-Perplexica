"""
Writing Assistant Agent Module

This module provides functionality for assisting users with their writing tasks.
It focuses on providing direct writing assistance without performing web searches.
"""

from typing import List, Dict, Any, AsyncGenerator, Callable, Optional
from langchain.schema import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import StrOutputParser
from backend.utils.logger import logger

WRITING_ASSISTANT_PROMPT = """
You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are currently set on focus mode 'Writing Assistant', this means you will be helping the user write a response to a given query. 
Since you are a writing assistant, you would not perform web searches. If you think you lack information to answer the query, you can ask the user for more information or suggest them to switch to a different focus mode.
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

async def create_writing_assistant_chain(llm: BaseChatModel) -> RunnableSequence:
    """
    Create the writing assistant chain.
    
    Args:
        llm: The language model to use for writing assistance
        
    Returns:
        RunnableSequence: A chain that processes user queries and generates writing assistance
    """
    chain = LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_messages([
            ("system", WRITING_ASSISTANT_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])
    )
    str_parser = StrOutputParser()
    
    async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a response to the user's writing-related query."""
        try:
            response = await chain.arun(
                query=input_data["query"],
                chat_history=input_data["chat_history"]
            )
            
            return {
                "type": "response",
                "data": str_parser.parse(response)
            }
        except Exception as e:
            logger.error(f"Error in writing assistant chain: {str(e)}")
            return {
                "type": "error",
                "data": "An error occurred while processing your writing request"
            }
    
    return RunnableSequence([generate_response])

async def basic_writing_assistant(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Run the writing assistant and yield results as they become available.
    
    Args:
        query: The user's writing-related query
        history: The chat history
        llm: The language model to use
    """
    try:
        writing_assistant_chain = await create_writing_assistant_chain(llm)
        
        result = await writing_assistant_chain.invoke({
            "query": query,
            "chat_history": history
        })
        
        yield result
        
    except Exception as e:
        logger.error(f"Error in writing assistant: {str(e)}")
        yield {
            "type": "error",
            "data": "An error has occurred please try again later"
        }

async def handle_writing_assistant(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Optional[Embeddings] = None  # Keep for compatibility but don't use
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle a writing assistance request and generate streaming responses.

    Args:
        query: The user's writing-related query or request
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: Not used, kept for compatibility

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    async for result in basic_writing_assistant(query, history, llm):
        yield result
