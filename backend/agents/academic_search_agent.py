"""
Academic Search Agent Module

This module provides functionality for performing academic searches using multiple academic search
engines (arXiv, Google Scholar, PubMed) and processing the results using a language model to
generate concise, well-cited summaries.
"""

from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel, RunnableLambda
from lib.searxng import search_searxng
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

BASIC_ACADEMIC_SEARCH_RETRIEVER_PROMPT = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return `not_needed` as the response.

Example:
1. Follow up question: How does stable diffusion work?
Rephrased: Stable diffusion working

2. Follow up question: What is linear algebra?
Rephrased: Linear algebra

3. Follow up question: What is the third law of thermodynamics?
Rephrased: Third law of thermodynamics

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

class AcademicSearchAgent(AbstractAgent):
    """Agent for performing academic searches and generating responses."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the academic search retriever chain."""
        prompt = PromptTemplate.from_template(BASIC_ACADEMIC_SEARCH_RETRIEVER_PROMPT)

        retriever_chain = prompt | self.llm

        async def process_llm_output(x: Dict[str, Any]) -> Dict[str, Any]:
            """Process the LLM output and perform the academic search."""
            rephrased_query = x["rephrased_query"]
            if isinstance(rephrased_query, str):
                if rephrased_query.strip() == "not_needed":
                    return {"query": "", "docs": []}
            else:
                rephrased_query = await rephrased_query
                if rephrased_query.strip() == "not_needed":
                    return {"query": "", "docs": []}

            try:
                search_results = await search_searxng(rephrased_query, {
                    "language": "en",
                    "engines": ["arxiv", "google scholar", "pubmed"]
                })

                documents = [
                    Document(
                        page_content=result["content"],
                        metadata={
                            "title": result["title"],
                            "url": result["url"],
                            "img_src": result.get("img_src")
                        }
                    )
                    for result in search_results["results"]
                ]

                return {"query": rephrased_query, "docs": documents}
            except Exception as e:
                logger.error("Error in academic search: %s", str(e))
                return {"query": rephrased_query, "docs": []}

        chain = RunnableSequence(
            RunnableParallel({
                "rephrased_query": retriever_chain,
                "original_input": RunnablePassthrough()
            }),
            RunnableLambda(process_llm_output)
        )

        return chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the academic search answering chain."""
        retriever_chain = await self.create_retriever_chain()

        async def process_retriever_output(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process the retriever output and generate the final response."""
            try:
                retriever_result = await retriever_chain.ainvoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
                })

                if not retriever_result["query"]:
                    return {
                        "response": "I'm not sure how to help with that. Could you please ask a specific question?",
                        "sources": []
                    }

                reranked_docs = await self.rerank_docs(
                    retriever_result["query"],
                    retriever_result["docs"]
                )

                context = await self.process_docs(reranked_docs)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.format_prompt(retriever_result["query"], context)),
                    ("human", "{query}"),
                ])

                response_chain = prompt | self.llm

                response = await response_chain.ainvoke({
                    "query": retriever_result["query"],
                    "context": context
                })

                return {
                    "response": response.content,
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
                error_msg = "An error occurred while processing the academic search results"
                if "rate limit" in str(e).lower():
                    error_msg = "Rate limit exceeded. Please try again later."
                return {
                    "response": error_msg,
                    "sources": []
                }

        return RunnableSequence(
            RunnablePassthrough(),
            RunnableLambda(process_retriever_output)
        )

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        base_prompt = self.create_base_prompt()
        return f"""
        {base_prompt}
        
        You are set on focus mode 'Academic', this means you will be searching for academic papers and articles on the web.
        The context provided contains academic search results from sources like arXiv, Google Scholar, and PubMed.
        
        <context>
        {context}
        </context>
        """

async def handle_academic_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle an academic search query and generate streaming responses.

    Args:
        query: The academic search query
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    agent = AcademicSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result
