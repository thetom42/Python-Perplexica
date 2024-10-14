"""
Wolfram Alpha Search Agent Module

This module provides functionality for performing searches using the Wolfram Alpha API
and interpreting the results using a language model to generate clear and concise explanations.
"""

from langchain.schema import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from typing import List, Dict, Any
from backend.config import get_wolfram_alpha_app_id

async def handle_wolfram_alpha_search(query: str, history: List[BaseMessage], llm: BaseChatModel, embeddings: Embeddings) -> Dict[str, Any]:
    """
    Handle a Wolfram Alpha search query and generate a response based on the results.

    This function performs the following steps:
    1. Executes a query using the Wolfram Alpha API.
    2. Processes the Wolfram Alpha result.
    3. Uses a language model to generate an explanation of the Wolfram Alpha output.

    Args:
        query (str): The Wolfram Alpha search query.
        history (List[BaseMessage]): The chat history (not used in the current implementation).
        llm (BaseChatModel): The language model to use for generating the response.
        embeddings (Embeddings): The embeddings model (not used in the current implementation).

    Returns:
        Dict[str, Any]: A dictionary containing the type of response, the generated explanation,
                        and the raw Wolfram Alpha result.
    """
    wolfram_alpha_app_id = get_wolfram_alpha_app_id()
    wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid=wolfram_alpha_app_id)

    try:
        wolfram_result = wolfram.run(query)
    except Exception as e:
        return {
            "type": "error",
            "data": f"Error querying Wolfram Alpha: {str(e)}"
        }

    response_generator = LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_messages([
            ("system", """
            You are an AI assistant specializing in interpreting Wolfram Alpha results.
            Explain the Wolfram Alpha output in a clear and concise manner.
            If the result contains mathematical or scientific information, provide additional context or explanations.
            If the query couldn't be answered, suggest how to rephrase or clarify the question.
            """),
            ("human", "{query}"),
            ("human", "Wolfram Alpha result:\n{wolfram_result}")
        ])
    )

    response = await response_generator.arun(query=query, wolfram_result=wolfram_result)

    return {
        "type": "response",
        "data": response,
        "wolfram_result": wolfram_result
    }

# Additional helper functions can be implemented as needed
