"""
Wolfram Alpha Search Agent Module

This module provides functionality for performing Wolfram Alpha searches using the SearxNG search engine.
"""

from datetime import datetime
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

        chain = prompt | self.llm | str_parser

        async def process_output(query_text: str) -> Dict[str, Any]:
            """Process the LLM output and perform the Wolfram Alpha search."""
            if query_text.strip() == "not_needed":
                return {"query": "", "docs": []}

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

                return {"query": query_text, "docs": documents}
            except Exception as e:
                logger.error("Error in Wolfram Alpha search: %s", str(e))
                return {"query": query_text, "docs": []}

        retriever_chain = RunnableSequence([
            {
                "rephrased_query": chain,
                "original_input": RunnablePassthrough()
            },
            lambda x: process_output(x["rephrased_query"])
        ])

        return retriever_chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the Wolfram Alpha search answering chain."""
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

                # No reranking needed for Wolfram Alpha results as they are already highly relevant
                docs = retriever_result["docs"]
                context = await self.process_docs(docs)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.format_prompt(input_data["query"], context)),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{query}")
                ])

                response_chain = prompt | self.llm

                response = await response_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"]
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
                        for doc in docs
                    ]
                }
            except Exception as e:
                logger.error("Error in response generation: %s", str(e))
                return {
                    "type": "error",
                    "data": "An error occurred while generating the response"
                }

        return RunnableSequence([generate_response])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        return f"""
        You are Perplexica, an AI model who is expert at searching the web and answering user's queries. You are set on focus mode 'Wolfram Alpha', this means you will be searching for information on the web using Wolfram Alpha. It is a computational knowledge engine that can answer factual queries and perform computations.

        Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containing a brief description of the content of that page).
        You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
        You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
        Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
        You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
        Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
        However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.

        Anything inside the following `context` HTML block provided below is for your knowledge returned by Wolfram Alpha and is not shared by the user. You have to answer question on the basis of it and cite the relevant information from it but you do not have to 
        talk about the context in your response. 

        <context>
        {context}
        </context>

        If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
        Anything between the `context` is retrieved from Wolfram Alpha and is not a part of the conversation with the user. Today's date is {datetime.now().isoformat()}
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
