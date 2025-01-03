"""
Abstract Agent Base Class

This module provides the base class for all search agents, defining common interfaces
and shared functionality for processing queries and generating responses.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Literal, AsyncGenerator, Optional
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableSequence
from utils.compute_similarity import compute_similarity
from utils.logger import logger

class AbstractAgent(ABC):
    """Abstract base class for all search agents."""

    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Optional[Embeddings] = None,
        optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
    ):
        """
        Initialize the abstract agent.

        Args:
            llm: The language model to use
            embeddings: Optional embeddings model for document reranking
            optimization_mode: The mode determining the trade-off between speed and quality
        """
        self.llm = llm
        self.embeddings = embeddings
        self.optimization_mode = optimization_mode

    @abstractmethod
    async def create_retriever_chain(self) -> RunnableSequence:
        """
        Create the retriever chain for the agent.

        This method should be implemented by concrete classes to define
        how queries are processed and documents are retrieved.
        """

    @abstractmethod
    async def create_answering_chain(self) -> RunnableSequence:
        """
        Create the answering chain for the agent.

        This method should be implemented by concrete classes to define
        how responses are generated from retrieved documents.
        """

    async def process_docs(self, docs: List[Document]) -> str:
        """
        Process documents into a numbered string format.

        Args:
            docs: List of documents to process

        Returns:
            str: Formatted string with numbered documents
        """
        return "\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs)])

    async def rerank_docs(
        self,
        query: str,
        docs: List[Document]
    ) -> List[Document]:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: The search query
            docs: List of documents to rerank

        Returns:
            List[Document]: Reranked documents
        """
        try:
            if not docs:
                return []

            docs_with_content = [doc for doc in docs if doc.page_content and len(doc.page_content) > 0]

            if self.optimization_mode == "speed" or not self.embeddings:
                return docs_with_content[:15]
            elif self.optimization_mode == "balanced":
                doc_embeddings = await self.embeddings.embed_documents([doc.page_content for doc in docs_with_content])
                query_embedding = await self.embeddings.embed_query(query)

                similarity_scores = [
                    {"index": i, "similarity": compute_similarity(query_embedding, doc_embedding)}
                    for i, doc_embedding in enumerate(doc_embeddings)
                ]

                # Filter docs with similarity > 0.3 and take top 15
                sorted_docs = sorted(similarity_scores, key=lambda x: x["similarity"], reverse=True)
                filtered_docs = [docs_with_content[score["index"]] for score in sorted_docs if score["similarity"] > 0.3][:15]
                return filtered_docs

            return docs_with_content  # Default case for "quality" mode
        except Exception as e:
            logger.error("Error in document reranking: %s", str(e))
            return docs[:15]  # Fallback to simple truncation

    async def basic_search(
        self,
        query: str,
        history: List[BaseMessage]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Perform a search and yield results as they become available.

        Args:
            query: The search query
            history: The chat history

        Yields:
            Dict[str, Any]: Dictionaries containing response types and data
        """
        try:
            answering_chain = await self.create_answering_chain()
            
            result = await answering_chain.ainvoke({
                "query": query,
                "chat_history": history
            })

            if isinstance(result, dict):
                if "error" in result:
                    yield {
                        "type": "error",
                        "data": result["error"]
                    }
                elif "sources" in result:
                    yield {"type": "sources", "data": result["sources"]}
                yield {"type": "response", "data": result.get("response", result.get("data", ""))}
            else:
                yield {"type": "response", "data": str(result)}

        except Exception as e:
            logger.error("Error in search: %s", str(e))
            yield {
                "type": "error",
                "data": f"An error occurred: {str(e)}"
            }

    async def handle_search(
        self,
        query: str,
        history: List[BaseMessage]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle a search query and generate streaming responses.

        Args:
            query: The search query
            history: The chat history

        Yields:
            Dict[str, Any]: Dictionaries containing response types and data
        """
        try:
            async for result in self.basic_search(query, history):
                yield result
        except Exception as e:
            logger.error(f"Error in search handling: {str(e)}")
            yield {
                "type": "error",
                "data": f"An error occurred: {str(e)}"
            }

    def create_base_prompt(self) -> str:
        """
        Create the base prompt template for all agents.
        
        Returns:
            str: The base prompt template
        """
        return """
        You are Perplexica, an AI model who is expert at searching the web and answering user's queries.
        
        Generate a response that is informative and relevant to the user's query based on provided context.
        You must use this context to answer the user's query in the best way possible. Use an unbiased and journalistic tone in your response.
        Your responses should be medium to long in length and informative. You can use markdown to format your response.
        You have to cite the answer using [number] notation. Place these citations at the end of relevant sentences.
        """

    def format_prompt(self, query: str, context: str) -> str:
        """
        Format the prompt for the language model.

        Args:
            query: The user's query
            context: The context from retrieved documents

        Returns:
            str: The formatted prompt
        """
        base_prompt = self.create_base_prompt()
        return f"""
        {base_prompt}
        
        <context>
        {context}
        </context>
        
        Today's date is {datetime.now().isoformat()}
        """

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the response from the language model.

        Args:
            response: The raw response from the language model

        Returns:
            Dict[str, Any]: The parsed response with appropriate structure
        """
        return {
            "type": "response",
            "data": response
        }
