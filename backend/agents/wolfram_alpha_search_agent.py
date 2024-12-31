"""
Wolfram Alpha Search Agent Module

This module provides functionality for performing Wolfram Alpha searches using the SearxNG search engine.
"""

from typing import List, Dict, Any, Literal, AsyncGenerator, Optional
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from lib.searxng import search_searxng
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

BASIC_WOLFRAM_ALPHA_SEARCH_RETRIEVER_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return `not_needed` as the response.

Example:
1. Follow up question: What is the atomic radius of S?
Rephrased: Atomic radius of S

2. Follow up question: What is linear algebra?
Rephrased: Linear algebra

3. Follow up question: What is the third law of thermodynamics?
Rephrased: Third law of thermodynamics

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

class WolframAlphaSearchAgent(AbstractAgent):
    """Agent for performing Wolfram Alpha searches and generating responses."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the Wolfram Alpha search retriever chain."""
        prompt = PromptTemplate.from_template(BASIC_WOLFRAM_ALPHA_SEARCH_RETRIEVER_PROMPT)
        str_parser = StrOutputParser()

        async def process_output(query_text: str) -> Dict[str, Any]:
            """Process the LLM output and perform the Wolfram Alpha search."""
            # Handle the case where query_text is a coroutine
            if hasattr(query_text, '__await__'):
                query_text = await query_text

            # Convert to string and validate
            query_text = str(query_text)
            
            # Ensure query_text is a valid string
            if not isinstance(query_text, str):
                query_text = str(query_text)
                if not isinstance(query_text, str):
                    raise ValueError("Query text must be a valid string")

            if query_text.strip() == "not_needed":
                return {
                    "query": "",
                    "docs": []
                }

            # Validate query_text is not empty
            if not query_text.strip():
                raise ValueError("Query text cannot be empty")

            try:
                res = await search_searxng(query_text, {
                    "language": "en",
                    "engines": ["wolframalpha"]
                })

                documents = [
                    Document(
                        page_content=result["content"],
                        metadata={
                            "title": result["title"],
                            "url": result["url"],
                            **({"img_src": result["img_src"]} if "img_src" in result else {})
                        }
                    )
                    for result in res["results"]
                ]

                return {
                    "query": query_text,
                    "docs": documents
                }
            except Exception as e:
                logger.error("Error in Wolfram Alpha search: %s", str(e))
                raise

        # Create the initial chain to get the rephrased query
        initial_chain = prompt | self.llm | str_parser

        # Create a chain that combines the rephrased query with the original input
        combined_chain = RunnablePassthrough.assign(
            rephrased_query=lambda x: initial_chain.ainvoke(x)
        )

        # Create the final chain that processes the output
        async def process_chain(x):
            # Get the raw string response from the LLM
            query_text = await x["rephrased_query"]
            if hasattr(query_text, '__await__'):
                query_text = await query_text
            # Handle Generation object if present
            if hasattr(query_text, 'text'):
                query_text = query_text.text
            query_text = str(query_text)
            
            if query_text.strip() == "not_needed":
                return {
                    "query": "",
                    "docs": []
                }

            # Validate query_text is not empty
            if not query_text.strip():
                raise ValueError("Query text cannot be empty")

            try:
                res = await search_searxng(query_text, {
                    "language": "en",
                    "engines": ["wolframalpha"]
                })

                documents = [
                    Document(
                        page_content=result["content"],
                        metadata={
                            "title": result["title"],
                            "url": result["url"],
                            **({"img_src": result["img_src"]} if "img_src" in result else {})
                        }
                    )
                    for result in res["results"]
                ]

                return {
                    "query": query_text,
                    "docs": documents
                }
            except Exception as e:
                logger.error("Error in Wolfram Alpha search: %s", str(e))
                raise

        final_chain = combined_chain | process_chain

        return final_chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the Wolfram Alpha search answering chain."""
        retriever_chain = await self.create_retriever_chain()
        str_parser = StrOutputParser()

        async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate the final response using the retriever chain and response chain."""
            # Validate input types
            if not isinstance(input_data, dict):
                raise TypeError("Input data must be a dictionary")
            if not isinstance(input_data.get("query", ""), str):
                raise TypeError("Query must be a string")
            if not isinstance(input_data.get("chat_history", []), list):
                raise TypeError("Chat history must be a list")

            try:
                retriever_result = await retriever_chain.ainvoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                if not retriever_result["query"]:
                    return {
                        "type": "response",
                        "data": "I'm not sure how to help with that. Could you please ask a specific question?"
                    }

                # No reranking needed for Wolfram Alpha results as they are already highly relevant
                docs = retriever_result["docs"]
                context = await self.process_docs(docs)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.format_prompt(input_data["query"], context)),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{query}")
                ])

                response_chain = prompt | self.llm | str_parser
                response = await response_chain.ainvoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                # Handle the case where response is a coroutine
                if hasattr(response, '__await__'):
                    response = await response

                # Convert response to string if it's not already
                response_text = str(response)

                # Validate response type
                if not response_text or response_text.lower() == "not_needed":
                    return {
                        "type": "response",
                        "data": "I'm not sure how to help with that. Could you please ask a specific question?"
                    }

                # Ensure response_text is a valid string
                if not isinstance(response_text, str):
                    response_text = str(response_text)
                    if not isinstance(response_text, str):
                        raise ValueError("Response must be a valid string")

                return {
                    "type": "response",
                    "data": response_text,
                    "sources": [
                        {
                            "title": doc.metadata.get("title", ""),
                            "url": doc.metadata.get("url", ""),
                            "img_src": doc.metadata.get("img_src")
                        }
                        for doc in docs
                    ]
                }
            except Exception as e:
                logger.error("Error in response generation: %s", str(e))
                raise

        # Create a chain that processes the input and generates the response
        return RunnablePassthrough() | generate_response

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        base_prompt = self.create_base_prompt()
        return f"""
        {base_prompt}
        
        You are set on focus mode 'Wolfram Alpha', this means you will be searching for information using Wolfram Alpha,
        a computational knowledge engine that can answer factual queries and perform computations.
        
        <context>
        {context}
        </context>
        """

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response
        }

async def handle_wolfram_alpha_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Optional[Embeddings] = None,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle a Wolfram Alpha search query and generate streaming responses.

    Args:
        query: The Wolfram Alpha search query
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: Optional embeddings model (unused for Wolfram Alpha search)
        optimization_mode: The optimization mode (unused for Wolfram Alpha search)

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = WolframAlphaSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result
