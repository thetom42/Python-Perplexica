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
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.output_parsers import StrOutputParser
from backend.lib.searxng import search_searxng
from backend.utils.logger import logger
from backend.agents.abstract_agent import AbstractAgent, RunnableSequence

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
        chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(BASIC_YOUTUBE_SEARCH_RETRIEVER_PROMPT)
        )
        str_parser = StrOutputParser()
        
        async def process_input(input_data: Dict[str, Any]) -> str:
            """Process the input through the LLM chain."""
            try:
                return await chain.arun(
                    chat_history=input_data["chat_history"],
                    query=input_data["query"]
                )
            except Exception as e:
                logger.error(f"Error in LLM processing: {str(e)}")
                return input_data["query"]
        
        async def process_output(input: str) -> Dict[str, Any]:
            """Process the LLM output and perform the YouTube search."""
            if input.strip() == "not_needed":
                return {"query": "", "docs": []}
                
            try:
                res = await search_searxng(input, {
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
                
                return {"query": input, "docs": documents}
            except Exception as e:
                logger.error(f"Error in YouTube search: {str(e)}")
                return {"query": input, "docs": []}
        
        return RunnableSequence([process_input, str_parser.parse, process_output])

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the YouTube search answering chain."""
        retriever_chain = await self.create_retriever_chain()
        
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
                
                reranked_docs = await self.rerank_docs(retriever_result["query"], retriever_result["docs"])
                context = await self.process_docs(reranked_docs)
                
                response_chain = LLMChain(
                    llm=self.llm,
                    prompt=ChatPromptTemplate.from_messages([
                        ("system", self.format_prompt(input_data["query"], context)),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{query}")
                    ])
                )
                
                response = await response_chain.arun(
                    query=input_data["query"],
                    chat_history=input_data["chat_history"]
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
                    "data": "An error occurred while generating the response"
                }
        
        return RunnableSequence([generate_response])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        return f"""
        You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are set on focus mode 'Youtube', this means you will be searching for videos on the web using Youtube and providing information based on the video's transcript.

        Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
        You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
        You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
        Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
        You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
        Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].

        <context>
        {context}
        </context>

        If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
        Anything between the `context` is retrieved from Youtube and is not a part of the conversation with the user. Today's date is {datetime.now().isoformat()}
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
