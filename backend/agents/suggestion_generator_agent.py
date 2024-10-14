"""
Suggestion Generator Agent Module

This module provides functionality for generating relevant and diverse suggestions
based on a user's query and chat history using a language model.
"""

from langchain.schema import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any

async def handle_suggestion_generation(query: str, history: List[BaseMessage], llm: BaseChatModel, embeddings: Embeddings) -> Dict[str, Any]:
    """
    Generate suggestions based on the user's query and chat history.

    This function performs the following steps:
    1. Creates a language model chain with a specific prompt for generating suggestions.
    2. Runs the chain with the formatted chat history and current query.
    3. Returns the generated suggestions.

    Args:
        query (str): The current user query.
        history (List[BaseMessage]): The chat history.
        llm (BaseChatModel): The language model to use for generating suggestions.
        embeddings (Embeddings): The embeddings model (not used in the current implementation).

    Returns:
        Dict[str, Any]: A dictionary containing the type of response and the generated suggestions.
    """
    suggestion_generator = LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_messages([
            ("system", """
            You are an AI suggestion generator. Based on the user's query and chat history,
            generate a list of 5 relevant and diverse suggestions for further exploration or discussion.
            These suggestions should be related to the topic but offer new angles or deeper insights.
            Format your response as a numbered list.
            """),
            ("human", "Chat history:\n{history}\n\nCurrent query: {query}\n\nGenerate 5 suggestions:"),
        ])
    )

    suggestions = await suggestion_generator.arun(history=format_chat_history(history), query=query)

    return {
        "type": "suggestions",
        "data": suggestions
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
