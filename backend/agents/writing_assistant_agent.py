"""
Writing Assistant Agent Module

This module provides functionality for assisting users with their writing tasks,
offering suggestions, improvements, and feedback using a language model.
"""

from langchain.schema import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any

async def handle_writing_assistance(query: str, history: List[BaseMessage], llm: BaseChatModel, embeddings: Embeddings) -> Dict[str, Any]:
    """
    Handle a writing assistance request and generate a response based on the user's query and chat history.

    This function performs the following steps:
    1. Creates a language model chain with a specific prompt for writing assistance.
    2. Runs the chain with the formatted chat history and current query.
    3. Returns the generated writing assistance.

    Args:
        query (str): The user's writing-related query or request.
        history (List[BaseMessage]): The chat history.
        llm (BaseChatModel): The language model to use for generating the response.
        embeddings (Embeddings): The embeddings model (not used in the current implementation).

    Returns:
        Dict[str, Any]: A dictionary containing the type of response and the generated writing assistance.
    """
    writing_assistant = LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_messages([
            ("system", """
            You are an AI writing assistant. Your task is to help users improve their writing,
            provide suggestions, and offer constructive feedback. Based on the user's query and chat history,
            provide appropriate writing assistance. This may include:
            1. Suggesting improvements for grammar, style, and clarity
            2. Offering alternative phrasings or word choices
            3. Providing structure suggestions for essays or articles
            4. Helping with brainstorming ideas or overcoming writer's block
            5. Explaining writing concepts or techniques

            Tailor your response to the specific needs expressed in the user's query.
            Be encouraging and constructive in your feedback.
            """),
            ("human", "Chat history:\n{history}\n\nCurrent query: {query}\n\nProvide writing assistance:"),
        ])
    )

    assistance = await writing_assistant.arun(history=format_chat_history(history), query=query)

    return {
        "type": "response",
        "data": assistance
    }

def format_chat_history(history: List[BaseMessage]) -> str:
    """
    Format the chat history into a string representation.

    This function takes the last 5 messages from the chat history and
    formats them into a string, with each message prefixed by its type.

    Args:
        history (List[BaseMessage]): The full chat history.

    Returns:
        str: A formatted string representation of the last 5 messages in the chat history.
    """
    return "\n".join([f"{msg.type}: {msg.content}" for msg in history[-5:]])  # Only use last 5 messages for context

# Additional helper functions can be implemented as needed
