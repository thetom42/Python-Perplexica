"""Reddit Search Agent Module

This module provides functionality for performing Reddit searches using the SearxNG search engine.
It includes document reranking capabilities to prioritize the most relevant Reddit discussions
and comments.
"""

from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from lib.searxng import search_searxng
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

BASIC_REDDIT_SEARCH_RETRIEVER_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return `not_needed` as the response.

Example:
1. Follow up question: Which company is most likely to create an AGI
Rephrased: Which company is most likely to create an AGI

2. Follow up question: Is Earth flat?
Rephrased: Is Earth flat?

3. Follow up question: Is there life on Mars?
Rephrased: Is there life on Mars?

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

class RedditSearchAgent(AbstractAgent):
    """Agent for performing Reddit searches and generating responses."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the Reddit search retriever chain."""
        prompt = PromptTemplate.from_template(BASIC_REDDIT_SEARCH_RETRIEVER_PROMPT)
        str_parser = StrOutputParser()

        chain = prompt | self.llm | str_parser

        async def process_output(output: Dict[str, Any]) -> Dict[str, Any]:
            """Process the LLM output and perform the Reddit search."""
            query_text = output.get("rephrased_query", "")
            if isinstance(query_text, str) and query_text.strip() == "not_needed":
                return {"query": "", "docs": []}

            try:
                res = await search_searxng(query_text, {
                    "language": "en",
                    "engines": ["reddit"]
                })

                documents = [
                    Document(
                        page_content=result.get("content", result["title"]),
                        metadata={
                            "title": result["title"],
                            "url": result["url"],
                            **({"img_src": result["img_src"]} if result.get("img_src") else {})
                        }
                    )
                    for result in res["results"]
                ]

                return {"query": query_text, "docs": documents}
            except Exception as e:
                logger.error("Error in reddit search: %s", str(e))
                return {"query": query_text, "docs": []}

        return RunnableSequence(
            RunnablePassthrough.assign(rephrased_query=chain),
            RunnableLambda(process_output)
        )

    async def handle_search(self, query: str, history: List[BaseMessage]) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle a search query and generate streaming responses."""
        answering_chain = await self.create_answering_chain()
        result = await answering_chain.ainvoke({
            "query": query,
            "chat_history": history
        })
        yield result

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the Reddit search answering chain."""
        def validate_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(input_data.get("query"), str) or not isinstance(input_data.get("chat_history"), list):
                raise TypeError("Invalid input types: query must be string and chat_history must be list")
            return input_data
            
        retriever_chain = await self.create_retriever_chain()

        async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate the final response using the retriever chain and response chain."""
            if not isinstance(input_data.get("query"), str) or not isinstance(input_data.get("chat_history"), list):
                raise TypeError("Invalid input types: query must be string and chat_history must be list")
            
            try:
                retriever_result = await retriever_chain.ainvoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })
            except Exception as e:
                error_msg = str(e)
                if "Rate limit exceeded" in error_msg:
                    return {
                        "type": "response",
                        "data": "I'm currently experiencing rate limits. Please try again in a few minutes.",
                        "sources": []
                    }
                logger.error("Error in response generation: %s", error_msg)
                return {
                    "type": "response",
                    "data": "An error occurred while processing your request. Please try again.",
                    "sources": []
                }

            if not retriever_result["query"]:
                return {
                    "type": "response",
                    "data": "I'm not sure how to help with that. Could you please ask a specific question?",
                    "sources": []
                }

            docs = retriever_result["docs"]
            if hasattr(docs, '__await__'):
                docs = await docs
            reranked_docs = await self.rerank_docs(retriever_result["query"], docs)
            context = await self.process_docs(reranked_docs)
            formatted_prompt = self.format_prompt(input_data["query"], context)

            try:
                response = await self.llm.ainvoke({"query": formatted_prompt})
                response_content = response.content if hasattr(response, 'content') else str(response)

                return {
                    "type": "response",
                    "data": response_content,
                    "sources": [
                        {
                            "title": doc.metadata.get("title", ""),
                            "url": doc.metadata.get("url", ""),
                            "img_src": doc.metadata.get("img_src")
                        }
                        for doc in reranked_docs
                    ]
                }
            except Exception as e:
                error_msg = str(e)
                if "Rate limit exceeded" in error_msg:
                    return {
                        "type": "response",
                        "data": "I'm currently experiencing high traffic. Please try again later.",
                        "sources": []
                    }
                logger.error("Error in response generation: %s", error_msg)
                return {
                    "type": "response",
                    "data": "An error occurred while generating the response. Please try again.",
                    "sources": []
                }

        return RunnableSequence(
            RunnablePassthrough(),
            RunnableLambda(generate_response)
        )

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        base_prompt = self.create_base_prompt()
        return f"""
        {base_prompt}
        
        You are set on focus mode 'Reddit', this means you will be searching for information, opinions and discussions on Reddit.
        The context provided contains search results from Reddit discussions and comments.
        
        <context>
        {context}
        </context>
        """

async def handle_reddit_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle a Reddit search query and generate streaming responses.

    Args:
        query: The Reddit search query
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = RedditSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result
