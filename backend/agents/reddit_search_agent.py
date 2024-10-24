"""
Reddit Search Agent Module

This module provides functionality for performing Reddit searches using the SearxNG search engine.
It includes document reranking capabilities to prioritize the most relevant Reddit discussions
and comments.
"""

from datetime import datetime
from typing import List, Dict, Any, Literal, AsyncGenerator, Callable
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import StrOutputParser
from backend.lib.searxng import search_searxng
from backend.utils.compute_similarity import compute_similarity
from backend.utils.format_history import format_chat_history_as_string
from backend.utils.logger import logger

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

BASIC_REDDIT_SEARCH_RESPONSE_PROMPT = f"""
You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are set on focus mode 'Reddit', this means you will be searching for information, opinions and discussions on the web using Reddit.

Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.

Anything inside the following `context` HTML block provided below is for your knowledge returned by Reddit and is not shared by the user. You have to answer question on the basis of it and cite the relevant information from it but you do not have to
talk about the context in your response.

<context>
{{context}}
</context>

If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
Anything between the `context` is retrieved from Reddit and is not a part of the conversation with the user. Today's date is {datetime.now().isoformat()}
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

async def create_basic_reddit_search_retriever_chain(llm: BaseChatModel) -> RunnableSequence:
    """
    Create the basic Reddit search retriever chain.
    
    Args:
        llm: The language model to use for query processing
        
    Returns:
        RunnableSequence: A chain that processes queries and retrieves Reddit content
    """
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(BASIC_REDDIT_SEARCH_RETRIEVER_PROMPT)
    )
    str_parser = StrOutputParser()
    
    async def process_input(input_data: Dict[str, Any]) -> str:
        """Process the input through the LLM chain."""
        try:
            return await chain.arun(
                chat_history=format_chat_history_as_string(input_data["chat_history"]),
                query=input_data["query"]
            )
        except Exception as e:
            logger.error(f"Error in LLM processing: {str(e)}")
            return input_data["query"]  # Fallback to original query
    
    async def process_output(input: str) -> Dict[str, Any]:
        """Process the LLM output and perform the Reddit search."""
        if input.strip() == "not_needed":
            return {"query": "", "docs": []}
            
        try:
            res = await search_searxng(input, {
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
            
            return {"query": input, "docs": documents}
        except Exception as e:
            logger.error(f"Error in Reddit search: {str(e)}")
            return {"query": input, "docs": []}
    
    return RunnableSequence([process_input, str_parser.parse, process_output])

async def create_basic_reddit_search_answering_chain(
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"]
) -> RunnableSequence:
    """
    Create the basic Reddit search answering chain.
    
    Args:
        llm: The language model to use
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality
    """
    retriever_chain = await create_basic_reddit_search_retriever_chain(llm)
    
    async def process_docs(docs: List[Document]) -> str:
        """Process documents into a formatted string."""
        return "\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs)])
    
    async def rerank_docs(data: Dict[str, Any]) -> List[Document]:
        """Rerank documents based on relevance to the query."""
        try:
            query, docs = data["query"], data["docs"]
            if not docs:
                return docs
                
            docs_with_content = [doc for doc in docs if doc.page_content and len(doc.page_content) > 0]
            
            if optimization_mode == "speed":
                return docs_with_content[:15]
            elif optimization_mode == "balanced":
                doc_embeddings = await embeddings.embed_documents([doc.page_content for doc in docs_with_content])
                query_embedding = await embeddings.embed_query(query)
                
                similarity_scores = [
                    {"index": i, "similarity": compute_similarity(query_embedding, doc_embedding)}
                    for i, doc_embedding in enumerate(doc_embeddings)
                ]
                
                return [
                    docs_with_content[score["index"]]
                    for score in sorted(
                        [s for s in similarity_scores if s["similarity"] > 0.3],
                        key=lambda x: x["similarity"],
                        reverse=True
                    )[:15]
                ]
                
            return docs_with_content
        except Exception as e:
            logger.error(f"Error in document reranking: {str(e)}")
            return docs[:15]  # Fallback to simple truncation
    
    async def generate_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the final response using the retriever chain and response chain."""
        try:
            retriever_result = await retriever_chain.invoke({
                "query": input_data["query"],
                "chat_history": input_data["chat_history"]
            })
            
            if not retriever_result["query"]:
                return {
                    "type": "response",
                    "data": "I'm not sure how to help with that. Could you please ask a specific question?"
                }
            
            reranked_docs = await rerank_docs(retriever_result)
            context = await process_docs(reranked_docs)
            
            response_chain = LLMChain(
                llm=llm,
                prompt=ChatPromptTemplate.from_messages([
                    ("system", BASIC_REDDIT_SEARCH_RESPONSE_PROMPT),
                    ("human", "{query}")
                ])
            )
            
            response = await response_chain.arun(query=input_data["query"], context=context)
            
            return {
                "type": "response",
                "data": response,
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
                "type": "error",
                "data": "An error occurred while processing Reddit search results"
            }
    
    return RunnableSequence([generate_response])

async def basic_reddit_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"]
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Perform a Reddit search and yield results as they become available.
    
    Args:
        query: The search query
        history: The chat history
        llm: The language model to use
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality
    """
    try:
        answering_chain = await create_basic_reddit_search_answering_chain(
            llm,
            embeddings,
            optimization_mode
        )
        
        result = await answering_chain.invoke({
            "query": query,
            "chat_history": history
        })
        
        if "sources" in result:
            yield {"type": "sources", "data": result["sources"]}
        
        yield {"type": "response", "data": result["data"]}
        
    except Exception as e:
        logger.error(f"Error in RedditSearch: {str(e)}")
        yield {
            "type": "error",
            "data": "An error has occurred please try again later"
        }

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
    async for result in basic_reddit_search(query, history, llm, embeddings, optimization_mode):
        yield result
