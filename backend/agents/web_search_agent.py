"""
Web Search Agent Module

This module provides functionality for performing web searches, rephrasing queries,
and generating responses based on search results or webpage content. It includes
specialized handling for webpage summarization and link-based content retrieval.
"""

from datetime import datetime
from typing import List, Dict, Any, Literal, AsyncGenerator, Callable, Optional, Union
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.output_parsers import StrOutputParser
from backend.lib.searxng import search_searxng
from backend.utils.compute_similarity import compute_similarity
from backend.utils.format_history import format_chat_history_as_string
from backend.lib.link_document import get_documents_from_links
from backend.utils.logger import logger
from backend.lib.output_parsers.line_output_parser import LineOutputParser
from backend.lib.output_parsers.list_line_output_parser import LineListOutputParser

BASIC_SEARCH_RETRIEVER_PROMPT = """
You are an AI question rephraser. You will be given a conversation and a follow-up question,  you will have to rephrase the follow up question so it is a standalone question and can be used by another LLM to search the web for information to answer it.
If it is a smple writing task or a greeting (unless the greeting contains a question after it) like Hi, Hello, How are you, etc. than a question then you need to return `not_needed` as the response (This is because the LLM won't need to search the web for finding information on this topic).
If the user asks some question from some URL or wants you to summarize a PDF or a webpage (via URL) you need to return the links inside the `links` XML block and the question inside the `question` XML block. If the user wants to you to summarize the webpage or the PDF you need to return `summarize` inside the `question` XML block in place of a question and the link to summarize in the `links` XML block.
You must always return the rephrased question inside the `question` XML block, if there are no links in the follow-up question then don't insert a `links` XML block in your response.

There are several examples attached for your reference inside the below `examples` XML block

<examples>
1. Follow up question: What is the capital of France
Rephrased question:`
<question>
Capital of france
</question>
`

2. Hi, how are you?
Rephrased question`
<question>
not_needed
</question>
`

3. Follow up question: What is Docker?
Rephrased question: `
<question>
What is Docker
</question>
`

4. Follow up question: Can you tell me what is X from https://example.com
Rephrased question: `
<question>
Can you tell me what is X?
</question>

<links>
https://example.com
</links>
`

5. Follow up question: Summarize the content from https://example.com
Rephrased question: `
<question>
summarize
</question>

<links>
https://example.com
</links>
`
</examples>

Anything below is the part of the actual conversation and you need to use conversation and the follow-up question to rephrase the follow-up question as a standalone question based on the guidelines shared above.

<conversation>
{chat_history}
</conversation>

Follow up question: {query}
Rephrased question:
"""

BASIC_WEB_SEARCH_RESPONSE_PROMPT = f"""
You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are also an expert at summarizing web pages or documents and searching for content in them.

Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
If the query contains some links and the user asks to answer from those links you will be provided the entire content of the page inside the `context` XML block. You can then use this content to answer the user's query.
If the user asks to summarize content from some links, you will be provided the entire content of the page inside the `context` XML block. You can then use this content to summarize the text. The content provided inside the `context` block will be already summarized by another model so you just need to use that content to answer the user's query.
Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.

Anything inside the following `context` HTML block provided below is for your knowledge returned by the search engine and is not shared by the user. You have to answer question on the basis of it and cite the relevant information from it but you do not have to
talk about the context in your response.

<context>
{{context}}
</context>

If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'. You do not need to do this for summarization tasks.
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

async def create_basic_web_search_retriever_chain(llm: BaseChatModel) -> RunnableSequence:
    """
    Create the basic web search retriever chain.
    
    Args:
        llm: The language model to use for query processing
        
    Returns:
        RunnableSequence: A chain that processes queries and retrieves web content
    """
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(BASIC_SEARCH_RETRIEVER_PROMPT)
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
            return "<question>not_needed</question>"  # Safe fallback
    
    async def process_output(input: str) -> Dict[str, Any]:
        """Process the LLM output and perform the web search or link processing."""
        try:
            links_parser = LineListOutputParser(key="links")
            question_parser = LineOutputParser(key="question")
            
            links = await links_parser.parse(input)
            question = await question_parser.parse(input)
            
            if question == "not_needed":
                return {"query": "", "docs": []}
                
            if links:
                if not question:
                    question = "summarize"
                    
                docs = []
                link_docs = await get_documents_from_links({"links": links})
                doc_groups = []
                
                for doc in link_docs:
                    url_doc = next(
                        (d for d in doc_groups if d.metadata["url"] == doc.metadata["url"] and d.metadata.get("totalDocs", 0) < 10),
                        None
                    )
                    
                    if not url_doc:
                        doc.metadata["totalDocs"] = 1
                        doc_groups.append(doc)
                    else:
                        url_doc.page_content += f"\n\n{doc.page_content}"
                        url_doc.metadata["totalDocs"] = url_doc.metadata.get("totalDocs", 0) + 1
                
                for doc in doc_groups:
                    try:
                        summary = await llm.agenerate([[
                            ("system", """
                            You are a web search summarizer, tasked with summarizing a piece of text retrieved from a web search. Your job is to summarize the 
                            text into a detailed, 2-4 paragraph explanation that captures the main ideas and provides a comprehensive answer to the query.
                            If the query is "summarize", you should provide a detailed summary of the text. If the query is a specific question, you should answer it in the summary.
                            
                            - **Journalistic tone**: The summary should sound professional and journalistic, not too casual or vague.
                            - **Thorough and detailed**: Ensure that every key point from the text is captured and that the summary directly answers the query.
                            - **Not too lengthy, but detailed**: The summary should be informative but not excessively long. Focus on providing detailed information in a concise format.
                            """),
                            ("human", f"""
                            <query>
                            {question}
                            </query>

                            <text>
                            {doc.page_content}
                            </text>

                            Make sure to answer the query in the summary.
                            """)
                        ]])
                        
                        docs.append(Document(
                            page_content=summary.generations[0][0].text,
                            metadata={
                                "title": doc.metadata["title"],
                                "url": doc.metadata["url"]
                            }
                        ))
                    except Exception as e:
                        logger.error(f"Error summarizing document: {str(e)}")
                        # Include original content if summarization fails
                        docs.append(doc)
                
                return {"query": question, "docs": docs}
            else:
                res = await search_searxng(question, {
                    "language": "en"
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
                
                return {"query": question, "docs": documents}
        except Exception as e:
            logger.error(f"Error in web search/processing: {str(e)}")
            return {"query": "", "docs": []}
    
    return RunnableSequence([process_input, str_parser.parse, process_output])

async def create_basic_web_search_answering_chain(
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"]
) -> RunnableSequence:
    """
    Create the basic web search answering chain.
    
    Args:
        llm: The language model to use
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality
    """
    retriever_chain = await create_basic_web_search_retriever_chain(llm)
    
    async def process_docs(docs: List[Document]) -> str:
        """Process documents into a formatted string."""
        return "\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs)])
    
    async def rerank_docs(data: Dict[str, Any]) -> List[Document]:
        """Rerank documents based on relevance to the query."""
        try:
            query, docs = data["query"], data["docs"]
            if not docs:
                return docs
                
            if query.lower() == "summarize":
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
                    ("system", BASIC_WEB_SEARCH_RESPONSE_PROMPT),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{query}")
                ])
            )
            
            response = await response_chain.arun(
                query=input_data["query"],
                chat_history=input_data["chat_history"],
                context=context
            )
            
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
                "data": "An error occurred while processing the web search results"
            }
    
    return RunnableSequence([generate_response])

async def basic_web_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"]
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Perform a web search and yield results as they become available.
    
    Args:
        query: The search query
        history: The chat history
        llm: The language model to use
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality
    """
    try:
        answering_chain = await create_basic_web_search_answering_chain(
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
        logger.error(f"Error in websearch: {str(e)}")
        yield {
            "type": "error",
            "data": "An error has occurred please try again later"
        }

async def handle_web_search(
    query: str,
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Embeddings,
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle a web search query and generate streaming responses.

    Args:
        query: The web search query
        history: The chat history
        llm: The language model to use for generating the response
        embeddings: The embeddings model for document reranking
        optimization_mode: The mode determining the trade-off between speed and quality

    Yields:
        Dict[str, Any]: Dictionaries containing response types and data
    """
    async for result in basic_web_search(query, history, llm, embeddings, optimization_mode):
        yield result
