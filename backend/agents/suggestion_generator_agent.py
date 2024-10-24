"""
Suggestion Generator Agent Module

This module provides functionality for generating relevant suggestions based on chat history.
It analyzes the conversation context to provide meaningful follow-up questions or topics.
"""

from typing import List, Dict, Any, Literal, AsyncGenerator, Optional
from langchain.schema import BaseMessage, Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from backend.utils.format_history import format_chat_history_as_string
from backend.lib.output_parsers.list_line_output_parser import LineListOutputParser
from backend.utils.logger import logger
from backend.agents.abstract_agent import AbstractAgent, RunnableSequence

SUGGESTION_GENERATOR_PROMPT = """
You are an AI suggestion generator for an AI powered search engine. You will be given a conversation below. You need to generate 4-5 suggestions based on the conversation. The suggestion should be relevant to the conversation that can be used by the user to ask the chat model for more information.
You need to make sure the suggestions are relevant to the conversation and are helpful to the user. Keep a note that the user might use these suggestions to ask a chat model for more information. 
Make sure the suggestions are medium in length and are informative and relevant to the conversation.

Provide these suggestions separated by newlines between the XML tags <suggestions> and </suggestions>. For example:

<suggestions>
Tell me more about SpaceX and their recent projects
What is the latest news on SpaceX?
Who is the CEO of SpaceX?
</suggestions>

Conversation:
{chat_history}
"""

class SuggestionGeneratorAgent(AbstractAgent):
    """Agent for generating suggestions based on chat history."""

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the suggestion generator retriever chain."""
        async def process_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process the input and return empty docs since we don't need retrieval."""
            return {"query": "", "docs": []}
        
        return RunnableSequence([process_input])

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the suggestion generator answering chain."""
        async def map_input(input_data: Dict[str, Any]) -> Dict[str, str]:
            """Map the input data to the format expected by the LLM chain."""
            return {
                "chat_history": format_chat_history_as_string(input_data["chat_history"])
            }
        
        # Set temperature to 0 for more consistent suggestions
        if hasattr(self.llm, "temperature"):
            self.llm.temperature = 0
            
        chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(SUGGESTION_GENERATOR_PROMPT)
        )
        
        async def run_chain(input_data: Dict[str, str]) -> str:
            """Run the LLM chain to generate suggestions."""
            try:
                return await chain.arun(**input_data)
            except Exception as e:
                logger.error(f"Error in suggestion generation chain: {str(e)}")
                return "<suggestions>\nUnable to generate suggestions at this time\n</suggestions>"
        
        output_parser = LineListOutputParser(key="suggestions")
        
        async def process_output(output: str) -> Dict[str, Any]:
            """Process the output into the expected format."""
            try:
                suggestions = await output_parser.parse(output)
                return {
                    "type": "response",
                    "data": suggestions
                }
            except Exception as e:
                logger.error(f"Error parsing suggestions: {str(e)}")
                return {
                    "type": "response",
                    "data": ["Unable to generate suggestions at this time"]
                }
        
        return RunnableSequence([
            map_input,
            run_chain,
            process_output
        ])

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        # Not used for suggestion generation as we use a fixed prompt
        return SUGGESTION_GENERATOR_PROMPT

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        return {
            "type": "response",
            "data": response.split("\n") if response else []
        }

async def generate_suggestions(
    history: List[BaseMessage],
    llm: BaseChatModel,
    embeddings: Optional[Embeddings] = None,  # Not used but kept for interface compatibility
    optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"  # Not used but kept for interface compatibility
) -> List[str]:
    """
    Generate suggestions based on chat history.

    Args:
        history: The chat history to generate suggestions from
        llm: The language model to use for generating suggestions
        embeddings: Not used for suggestion generation
        optimization_mode: Not used for suggestion generation

    Returns:
        List[str]: A list of generated suggestions. Returns a single error message
                  if suggestion generation fails.
    """
    try:
        # Create a dummy embeddings object if none provided since AbstractAgent requires it
        if embeddings is None:
            class DummyEmbeddings(Embeddings):
                async def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return [[0.0] for _ in texts]
                async def embed_query(self, text: str) -> List[float]:
                    return [0.0]
            embeddings = DummyEmbeddings()
            
        agent = SuggestionGeneratorAgent(llm, embeddings, optimization_mode)
        async for result in agent.handle_search("", history):  # Empty query since we only need history
            if result["type"] == "response":
                return result["data"]
        return ["Unable to generate suggestions at this time"]
    except Exception as e:
        logger.error(f"Error in generate_suggestions: {str(e)}")
        return ["Unable to generate suggestions at this time"]
