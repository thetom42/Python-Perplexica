"""
Academic Search Agent Module

This module provides functionality for performing academic searches using multiple academic search
engines (arXiv, Google Scholar, PubMed) and processing the results using a language model to
generate concise, well-cited summaries.
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Literal, AsyncGenerator, Optional, Callable, Awaitable
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from backend.lib.searxng import search_searxng
from backend.utils.compute_similarity import compute_similarity
from backend.utils.format_history import format_chat_history_as_string
from backend.utils.logger import logger

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

BASIC_ACADEMIC_SEARCH_RESPONSE_PROMPT = f"""
You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are set on focus mode 'Academic', this means you will be searching for academic papers and articles on the web.

Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.

Anything inside the following `context` HTML block provided below is for your knowledge returned by the search engine and is not shared by the user. You have to answer question on the basis of it and cite the relevant information from it but you do not have to 
talk about the context in your response. 

<context>
{{context}}
</context>

If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
Anything between the `context` is retrieved from a search engine and is not a part of the conversation with the user. Today's date is {datetime.now().isoformat()}
"""

class RunnableSequence:
    """A simple implementation of a runnable sequence similar to TypeScript's RunnableSequence."""
    
    def __init__(self, steps: List[Callable]):
        """Initialize the RunnableSequence with a list of callable steps."""
        self.steps = steps
    
    async def invoke(self, input_data: Dict[str, Any]) -> Any:
        """Execute the sequence of steps on the input data."""
        result = input_data
        for step in self.steps:
            result = await step(result)
        return result

async def create_basic_academic_search_retriever_chain(llm: BaseChatModel) -> RunnableSequence:
    """
    Create the basic academic search retriever chain.
    
    Args:
        llm: The language model to use for query rephrasing
        
    Returns:
        RunnableSequence: A chain that processes queries and retrieves academic documents
    """
    retriever_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(BASIC_ACADEMIC_SEARCH_RETRIEVER_PROMPT)
    )
    
    async def run_llm(input_data: Dict[str, Any]) -> str:
        """Process the input through the LLM chain."""
        try:
            return await retriever_chain.arun(
                chat_history=format_chat_history_as_string(input_data["chat_history"]),
                query=input_data["query"]
            )
        except Exception as e:
            logger.error(f"Error in LLM processing: {str(e)}")
            return "not_needed"  # Safe fallback
    
    async def process_llm_output(rephrased_query: str) -> Dict[str, Any]:
        """Process the LLM output and perform the academic search."""
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
            logger.error(f"Error in academic search: {str(e)}")
            return {"query": rephrased_query, "docs": []}
    
    return RunnableSequence([run_llm, process_llm_output])

async def process_docs(docs: List[Document]) -> str:
    """
    Process documents into a numbered string format.
    
    Args:
        docs: List of documents to process
        
    Returns:
        str: Formatted string with numbered documents
    """
    return "\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs)])

async def rerank_docs(
    query: str,
    docs: List[Document],
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"]
) -> List[Document]:
    """
    Rerank documents based on relevance to the query.
    
    Args:
        query: The search query
        docs: List of documents to rerank
        embeddings: Embeddings model for similarity computation
        optimization_mode: The mode determining the trade-off between speed and quality
        
    Returns:
        List[Document]: Reranked documents
    """
    try:
        if not docs:
            return []
        
        docs_with_content = [doc for doc in docs if doc.page_content and len(doc.page_content) > 0]
        
        if optimization_mode == "speed":
            return docs_with_content[:15]
        elif optimization_mode == "balanced":
            logger.info("Using balanced mode for document reranking")
            doc_embeddings = await embeddings.embed_documents([doc.page_content for doc in docs_with_content])
            query_embedding = await embeddings.embed_query(query)
            
            similarity_scores = [
                {"index": i, "similarity": compute_similarity(query_embedding, doc_embedding)}
                for i, doc_embedding in enumerate(doc_embeddings)
            ]
            
            sorted_docs = sorted(similarity_scores, key=lambda x: x["similarity"], reverse=True)[:15]
            return [docs_with_content[score["index"]] for score in sorted_docs]
        
        return docs_with_content  # Default case for "quality" mode
    except Exception as e:
        logger.error(f"Error in document reranking: {str(e)}")
        return docs[:15]  # Fallback to simple truncation

async def create_basic_academic_search_answering_chain(
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"]
) -> RunnableSequence:
    """
    Create the basic academic search answering chain.
    
    Args:
        llm: The language model to use
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality
    """
    retriever_chain = await create_basic_academic_search_retriever_chain(llm)
    
    async def process_retriever_output(retriever_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process the retriever output and generate the final response."""
        try:
            if not retriever_result["query"]:
                return {
                    "response": "I'm not sure how to help with that. Could you please ask a specific question?",
                    "sources": []
                }
            
            # Rerank documents
            reranked_docs = await rerank_docs(
                retriever_result["query"],
                retriever_result["docs"],
                embeddings,
                optimization_mode
            )
            
            # Process documents into context
            context = await process_docs(reranked_docs)
            
            # Generate response
            response_chain = LLMChain(
                llm=llm,
                prompt=ChatPromptTemplate.from_messages([
                    ("system", BASIC_ACADEMIC_SEARCH_RESPONSE_PROMPT),
                    ("human", "{query}"),
                ])
            )
            
            response = await response_chain.arun(
                query=retriever_result["query"],
                context=context
            )
            
            return {
                "response": response,
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
            logger.error(f"Error in response generation: {str(e)}")
            return {
                "response": "An error occurred while processing the academic search results",
                "sources": []
            }
    
    return RunnableSequence([
        lambda x: retriever_chain.invoke(x),
        process_retriever_output
    ])

async def basic_academic_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"]
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Perform an academic search and yield results as they become available.
    
    Args:
        query: The search query
        history: The chat history
        llm: The language model to use
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality
    """
    try:
        answering_chain = await create_basic_academic_search_answering_chain(
            llm,
            embeddings,
            optimization_mode
        )
        
        result = await answering_chain.invoke({
            "query": query,
            "chat_history": history
        })
        
        # Yield sources first
        yield {"type": "sources", "data": result["sources"]}
        
        # Then yield the response
        yield {"type": "response", "data": result["response"]}
        
    except Exception as e:
        logger.error(f"Error in academic search: {str(e)}")
        yield {
            "type": "error",
            "data": "An error has occurred please try again later"
        }

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
    async for result in basic_academic_search(query, history, llm, embeddings, optimization_mode):
        yield result
