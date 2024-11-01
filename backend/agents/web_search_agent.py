"""
Web Search Agent Module

This module provides functionality for performing web searches, rephrasing queries,
and generating responses based on search results or webpage content. It includes
specialized handling for webpage summarization and link-based content retrieval.
"""

from datetime import datetime
from typing import List, Dict, Any, Literal, AsyncGenerator
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from lib.searxng import search_searxng
from lib.link_document import get_documents_from_links
from utils.logger import logger
from lib.output_parsers.line_output_parser import LineOutputParser
from lib.output_parsers.list_line_output_parser import ListLineOutputParser
from agents.abstract_agent import AbstractAgent
from utils.compute_similarity import compute_similarity


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

class WebSearchAgent(AbstractAgent):
    """Agent for performing web searches and generating responses."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the web search retriever chain."""
        prompt = PromptTemplate.from_template(BASIC_SEARCH_RETRIEVER_PROMPT)

        # Set temperature to 0 for more precise rephrasing
        if hasattr(self.llm, "temperature"):
            self.llm.temperature = 0

        chain = prompt | self.llm

        async def process_output(llm_output: Any) -> Dict[str, Any]:
            """Process the LLM output and perform the web search."""
            try:
                output_text = llm_output.content if hasattr(llm_output, 'content') else str(llm_output)
                links_parser = ListLineOutputParser(key="links")
                question_parser = LineOutputParser(key="question")

                links = await links_parser.parse(output_text)
                question = await question_parser.parse(output_text)

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
                            summary_prompt = ChatPromptTemplate.from_messages([
                                ("system", """
                                You are a web search summarizer, tasked with summarizing a piece of text retrieved from a web search. Your job is to summarize the 
                                text into a detailed, 2-4 paragraph explanation that captures the main ideas and provides a comprehensive answer to the query.
                                If the query is "summarize", you should provide a detailed summary of the text. If the query is a specific question, you should answer it in the summary.

                                - **Journalistic tone**: The summary should sound professional and journalistic, not too casual or vague.
                                - **Thorough and detailed**: Ensure that every key point from the text is captured and that the summary directly answers the query.
                                - **Not too lengthy, but detailed**: The summary should be informative but not excessively long. Focus on providing detailed information in a concise format.

                                The text will be shared inside the `text` XML tag, and the query inside the `query` XML tag.

                                <example>
                                <text>
                                Docker is a set of platform-as-a-service products that use OS-level virtualization to deliver software in packages called containers. 
                                It was first released in 2013 and is developed by Docker, Inc. Docker is designed to make it easier to create, deploy, and run applications 
                                by using containers.
                                </text>

                                <query>
                                What is Docker and how does it work?
                                </query>

                                Response:
                                Docker is a revolutionary platform-as-a-service product developed by Docker, Inc., that uses container technology to make application 
                                deployment more efficient. It allows developers to package their software with all necessary dependencies, making it easier to run in 
                                any environment. Released in 2013, Docker has transformed the way applications are built, deployed, and managed.
                                </example>

                                Everything below is the actual data you will be working with. Good luck!

                                <query>
                                {query}
                                </query>

                                <text>
                                {text}
                                </text>

                                Make sure to answer the query in the summary.
                                """),
                                ("human", "{query}")
                            ])

                            summary_chain = summary_prompt | self.llm

                            summary = await summary_chain.invoke({
                                "query": question,
                                "text": doc.page_content
                            })

                            docs.append(Document(
                                page_content=summary.content,
                                metadata={
                                    "title": doc.metadata["title"],
                                    "url": doc.metadata["url"]
                                }
                            ))
                        except Exception as e:
                            logger.error("Error summarizing document: %s", str(e))
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
                logger.error("Error in web search/processing: %s", str(e))
                return {"query": "", "docs": []}

        retriever_chain = RunnableSequence([
            {
                "llm_output": chain,
                "original_input": RunnablePassthrough()
            },
            lambda x: process_output(x["llm_output"])
        ])

        return retriever_chain

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the web search answering chain."""
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

                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.format_prompt(input_data["query"], context)),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{query}")
                ])

                response_chain = prompt | self.llm

                response = await response_chain.invoke({
                    "query": input_data["query"],
                    "chat_history": input_data["chat_history"],
                    "context": context
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
                    "data": "An error occurred while processing the web search results"
                }

        return RunnableSequence([generate_response])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        return f"""
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
        {context}
        </context>

        If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'. You do not need to do this for summarization tasks.
        Anything between the `context` is retrieved from a search engine and is not a part of the conversation with the user. Today's date is {datetime.now().isoformat()}
        """

    async def rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank documents based on relevance to query."""
        if not docs:
            return docs

        if query.lower() == "summarize":
            return docs

        docs_with_content = [doc for doc in docs if doc.page_content and len(doc.page_content) > 0]

        if self.optimization_mode == "speed":
            return docs_with_content[:15]
        elif self.optimization_mode == "balanced":
            doc_embeddings = await self.embeddings.embed_documents([doc.page_content for doc in docs_with_content])
            query_embedding = await self.embeddings.embed_query(query)

            similarities = []
            for i, doc_embedding in enumerate(doc_embeddings):
                sim = compute_similarity(query_embedding, doc_embedding)
                similarities.append({"index": i, "similarity": sim})

            # Filter docs with similarity > 0.3 and take top 15
            sorted_docs = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
            filtered_docs = [docs_with_content[s["index"]] for s in sorted_docs if s["similarity"] > 0.3][:15]
            return filtered_docs

        return docs

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response
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
    agent = WebSearchAgent(llm, embeddings, optimization_mode)
    async for result in agent.handle_search(query, history):
        yield result
