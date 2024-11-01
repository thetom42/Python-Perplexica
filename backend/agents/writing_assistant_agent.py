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
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

WRITING_ASSISTANT_PROMPT = """
You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are currently set on focus mode 'Writing Assistant', this means you will be helping the user write a response to a given query.
Since you are a writing assistant, you would not perform web searches. If you think you lack information to answer the query, you can ask the user for more information or suggest them to switch to a different focus mode.
"""

class WritingAssistantAgent(AbstractAgent):
    """Agent for providing writing assistance without web searches."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the writing assistant retriever chain."""
        async def process_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process the input and return empty docs since we don't need retrieval."""
            return {"query": input_data["query"], "docs": []}

        return RunnableSequence([process_input])

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the writing assistant answering chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", WRITING_ASSISTANT_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])

        chain = prompt | self.llm
        str_parser = StrOutputParser()

        async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate a response to the user's writing-related query."""
            try:
                response = await chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                return {
                    "type": "response",
                    "data": str_parser.parse(response.content)
                }
            except Exception as e:
                logger.error("Error in writing assistant chain: %s", str(e))
                return {
                    "type": "error",
                    "data": "An error occurred while processing your writing request"
                }

        return RunnableSequence([generate_response])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        # Not used as we use a fixed prompt template in create_answering_chain
        return WRITING_ASSISTANT_PROMPT

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response
        }

async def handle_writing_assistant(
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
