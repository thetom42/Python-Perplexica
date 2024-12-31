"""
Suggestion Generator Agent Module

This module provides functionality for generating relevant suggestions based on chat history.
It analyzes the conversation context to provide meaningful follow-up questions or topics.
"""

from typing import List, Dict, Any, Literal, Optional
from langchain.schema import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda
from utils.format_history import format_chat_history_as_string
from lib.output_parsers.list_line_output_parser import ListLineOutputParser
from utils.logger import logger
from agents.abstract_agent import AbstractAgent

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

    def __init__(self, llm: BaseChatModel, embeddings: Embeddings, optimization_mode: Literal["speed", "balanced", "quality"] = "balanced"):
        """Initialize the agent with the given language model and embeddings."""
        if not isinstance(llm, BaseChatModel):
            raise TypeError("llm must be an instance of BaseChatModel")
        if not isinstance(embeddings, Embeddings):
            raise TypeError("embeddings must be an instance of Embeddings")
        super().__init__(llm, embeddings, optimization_mode)

    async def create_retriever_chain(self) -> RunnableSequence:
        """Create the suggestion generator retriever chain."""
        async def process_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process the input and return empty docs since we don't need retrieval."""
            return {"query": "", "docs": []}

        return RunnableSequence(
            RunnableLambda(lambda x: {"query": "", "docs": []}),
            RunnableLambda(process_input)
        )

    async def create_answering_chain(self) -> RunnableSequence:
        """Create the suggestion generator answering chain."""
        # Set temperature to 0 for more consistent suggestions
        if hasattr(self.llm, "temperature"):
            self.llm.temperature = 0

        prompt = PromptTemplate.from_template(SUGGESTION_GENERATOR_PROMPT)
        output_parser = ListLineOutputParser(key="suggestions")

        async def map_input(input_data: Dict[str, Any]) -> Dict[str, str]:
            """Map the input data to the format expected by the prompt."""
            return {
                "chat_history": format_chat_history_as_string(input_data["chat_history"])
            }

        async def process_output(output: str) -> Dict[str, Any]:
            """Process the output into the expected format."""
            try:
                suggestions = await output_parser.parse(output)
                return {
                    "type": "response",
                    "data": suggestions
                }
            except Exception as e:
                raise e

        chain = RunnableSequence(
            RunnableLambda(map_input),
            prompt | self.llm,
            RunnableLambda(lambda x: x.content if hasattr(x, 'content') else x),
            RunnableLambda(process_output)
        )

        return chain

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the language model."""
        base_prompt = self.create_base_prompt()
        return f"""
        {base_prompt}
        
        You are set on focus mode 'Suggestion Generation', this means you will analyze the conversation history
        to generate relevant follow-up questions or topics that the user might want to explore.
        """

    async def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the response from the language model."""
        if not response:
            return {
                "type": "response",
                "data": []
            }
        
        # Check if response is in valid format
        if not response.strip() or response.strip() == "Invalid format":
            raise ValueError("Invalid response format")
        
        # Extract content from XML tags if present
        if "<suggestions>" in response and "</suggestions>" in response:
            start = response.find("<suggestions>") + len("<suggestions>")
            end = response.find("</suggestions>")
            if start == -1 or end == -1 or start >= end:
                raise ValueError("Invalid response format")
            response = response[start:end]
        
        # Split into lines and clean up
        suggestions = [s.strip() for s in response.split("\n") if s.strip()]
        return {
            "type": "response",
            "data": suggestions
        }

async def handle_suggestion_generation(
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
    # Create a dummy embeddings object if none provided since AbstractAgent requires it
    if embeddings is None:
        class DummyEmbeddings(Embeddings):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [[0.0] for _ in texts]
            def embed_query(self, text: str) -> List[float]:
                return [0.0]
        embeddings = DummyEmbeddings()

    try:
        agent = SuggestionGeneratorAgent(llm, embeddings, optimization_mode)
        async for result in agent.handle_search("", history):  # Empty query since we only need history
            if result["type"] == "response":
                return result["data"]
        return ["Unable to generate suggestions at this time"]
    except Exception as e:
        # Log the error first with exact format
        logger.error("Error in suggestion generation: %s", str(e))
        # Then handle rate limit case
        error_msg = str(e).lower()
        if "rate limit" in error_msg or "ratelimit" in error_msg:
            return ["Unable to generate suggestions due to rate limit - please try again later"]
        # Default error case
        return ["Unable to generate suggestions at this time"]
