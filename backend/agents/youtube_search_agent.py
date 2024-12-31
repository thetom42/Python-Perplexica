"""
YouTube Search Agent Module

This module provides functionality for performing YouTube searches using the SearxNG search engine.
It includes document reranking capabilities for improved result relevance.
"""

from datetime import datetime
from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from lib.searxng import search_searxng
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

BASIC_YOUTUBE_SEARCH_RETRIEVER_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return `not_needed` as the response.

Example:
1. Follow up question: How does an A.C work?
Rephrased: A.C working

2. Follow up question: Linear algebra explanation video
Rephrased: What is linear algebra?

3. Follow up question: What is theory of relativity?
Rephrased: What is theory of relativity?

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

class YouTubeSearchAgent(AbstractAgent):
    """Agent for performing YouTube searches and generating responses."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the YouTube search retriever chain."""
        prompt = PromptTemplate.from_template(BASIC_YOUTUBE_SEARCH_RETRIEVER_PROMPT)
        str_parser = StrOutputParser()

        chain = RunnableParallel({
            "rephrased_query": prompt | self.llm | str_parser,
            "original_input": RunnablePassthrough()
        })

        async def process_output(data: Dict[str, Any]) -> Dict[str, Any]:
            """Process the LLM output and perform the YouTube search."""
            query_text = data["rephrased_query"]
            if not query_text or query_text.strip() == "not_needed":
                return {"query": "", "docs": []}
            
            # Ensure query_text is a string
            if not isinstance(query_text, str):
                query_text = str(query_text)

            try:
                res = await search_searxng(query_text, {
                    "language": "en",
                    "engines": ["youtube"]
                })

                documents = [
                    Document(
                        page_content=result.get("content", result["title"]),
                        metadata={
                            "title": result["title"],
                            "url": result["url"],
                            **({"img_src": result["img_src"]} if "img_src" in result else {})
                        }
                    )
                    for result in res["results"]
                ]

                return {"query": query_text, "docs": documents}
            except Exception as e:
                logger.error("Error in YouTube search: %s", str(e))
                return {"query": query_text, "docs": []}

        return RunnableSequence(chain, process_output)

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the YouTube search answering chain."""
        retriever_chain = await self.create_retriever_chain()

        async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate the final response using the retriever chain and response chain."""
            try:
                # Execute the entire chain with error handling
                retriever_result = await retriever_chain.ainvoke({
                    "query": input_data["query"],
                    "chat_history": input_data.get("chat_history", [])
                })

                if not retriever_result["query"]:
                    return {
                        "type": "response",
                        "data": "I'm not sure how to help with that. Could you please ask a specific question?",
                        "sources": []
                    }

                try:
                    reranked_docs = await self.rerank_docs(retriever_result["query"], retriever_result["docs"])
                    context = await self.process_docs(reranked_docs)

                    prompt = ChatPromptTemplate.from_messages([
                        ("system", self.format_prompt(input_data["query"], context)),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{query}")
                    ])

                    response_chain = prompt | self.llm

                    response = await response_chain.ainvoke({
                        "query": input_data["query"],
                        "chat_history": input_data.get("chat_history", [])
                    })

                    return {
                        "type": "response",
                        "data": response.content,
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
                    logger.error("Error in response generation: %s", str(e))
                    return {
                        "type": "error",
                        "data": f"An error occurred: {str(e)}",
                        "sources": [
                            {
                                "title": "Error",
                                "url": "",
                                "img_src": ""
                            }
                        ]
                    }
            except Exception as e:
                logger.error("Error in retriever chain: %s", str(e))
                return {
                    "type": "error",
                    "data": f"An error occurred: {str(e)}",
                    "sources": []
                }

        # Wrap the entire chain in error handling
        async def wrapped_chain(input_data: Dict[str, Any]) -> Dict[str, Any]:
            try:
                return await generate_response(input_data)
            except Exception as e:
                logger.error("Error in answering chain: %s", str(e))
                return {
                    "type": "error",
                    "data": f"An error occurred: {str(e)}",
                    "sources": []
                }
        
        return RunnableSequence(
            retriever_chain,
            wrapped_chain
        )

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        base_prompt = self.create_base_prompt()
        return f"""
        {base_prompt}
        
        Query: {query}
        
        You are set on focus mode 'YouTube', this means you will be searching for relevant videos on YouTube.
        The context provided contains video search results from YouTube.
        
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

async def handle_youtube_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle a YouTube search query and generate streaming responses.

    Args:
        query: The YouTube search query
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = YouTubeSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result
