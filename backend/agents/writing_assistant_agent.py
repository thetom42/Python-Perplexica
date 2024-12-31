"""
Writing Assistant Agent Module

This module provides functionality for assisting users with their writing tasks.
It focuses on providing direct writing assistance without performing web searches.
"""

from typing import List, Dict, Any, Literal, AsyncGenerator, Optional
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

class WritingAssistantAgent(AbstractAgent):
    """Agent for providing writing assistance without web searches."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the writing assistant retriever chain."""
        async def process_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process the input and return empty docs since we don't need retrieval."""
            return {"query": input_data["query"], "docs": []}

        return RunnableSequence(
            RunnableLambda(lambda x: x),  # Identity step
            RunnableLambda(process_input)
        )

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the writing assistant answering chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.format_prompt("", "")),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])

        # Create the chain with proper output handling
        chain = (
            prompt
            | self.llm
            | RunnableLambda(lambda x: x.text if hasattr(x, "text") else str(x))
        )

        async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate a response to the user's writing-related query."""
            try:
                response = await chain.ainvoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })
                
                # Handle different response types
                if isinstance(response, str):
                    return {
                        "type": "response",
                        "data": response
                    }
                elif hasattr(response, "text"):  # Handle Generation object
                    return {
                        "type": "response",
                        "data": response.text
                    }
                elif hasattr(response, "content"):
                    return {
                        "type": "response",
                        "data": response.content
                    }
                elif isinstance(response, dict):
                    if "error" in response:
                        return {
                            "type": "error",
                            "data": response["error"]
                        }
                    elif "response" in response:
                        return {
                            "type": "response",
                            "data": response["response"]
                        }
                return {
                    "type": "response",
                    "data": str(response)
                }
            except Exception as e:
                logger.error("Error in writing assistant chain: %s", str(e))
                return {
                    "type": "error",
                    "data": f"An error occurred: {str(e)}"
                }

        return RunnableSequence(
            RunnableLambda(lambda x: x),  # Identity step
            RunnableLambda(generate_response)
        )

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        base_prompt = self.create_base_prompt()
        return f"""
        {base_prompt}
        
        You are set on focus mode 'Writing Assistant', this means you will be helping the user with their writing tasks.
        Your role is to assist with writing, editing, and improving text rather than performing web searches.
        If you need more information to complete the writing task, ask the user for clarification or additional details.
        """

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        try:
            return {
                "type": "response",
                "data": response
            }
        except Exception as e:
            logger.error("Error parsing response: %s", str(e))
            return {
                "type": "error",
                "data": "An error occurred while processing the response. Please try again."
            }

async def handle_writing_assistance(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Optional[Embeddings] = None,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle a writing assistance request and generate streaming responses.

    Args:
        query: The user's writing-related query or request
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: Optional embeddings model, not used for writing assistance
        optimization_mode: Not used for writing assistance but required by base class

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = WritingAssistantAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result
