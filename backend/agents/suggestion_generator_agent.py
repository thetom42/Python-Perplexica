"""
Suggestion Generator Agent Module

This module provides functionality for generating relevant suggestions based on chat history.
It analyzes the conversation context to provide meaningful follow-up questions or topics.
"""

from typing import List, Dict, Any, Callable
from langchain.schema import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from backend.utils.format_history import format_chat_history_as_string
from backend.lib.output_parsers.list_line_output_parser import LineListOutputParser
from backend.utils.logger import logger

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

async def create_suggestion_generator_chain(llm: BaseChatModel) -> RunnableSequence:
    """
    Create the suggestion generator chain.
    
    Args:
        llm: The language model to use for generating suggestions
        
    Returns:
        RunnableSequence: A chain that processes chat history and generates suggestions
    """
    async def map_input(input_data: Dict[str, Any]) -> Dict[str, str]:
        """Map the input data to the format expected by the LLM chain."""
        return {
            "chat_history": format_chat_history_as_string(input_data["chat_history"])
        }
    
    chain = LLMChain(
        llm=llm,
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
    
    async def parse_output(output: str) -> List[str]:
        """Parse the LLM output into a list of suggestions."""
        try:
            return await output_parser.parse(output)
        except Exception as e:
            logger.error(f"Error parsing suggestions: {str(e)}")
            return ["Unable to generate suggestions at this time"]
    
    return RunnableSequence([
        map_input,
        run_chain,
        parse_output
    ])

async def generate_suggestions(
    history: List[BaseMessage],
    llm: BaseChatModel
) -> List[str]:
    """
    Generate suggestions based on chat history.

    Args:
        history: The chat history to generate suggestions from
        llm: The language model to use for generating suggestions

    Returns:
        List[str]: A list of generated suggestions. Returns a single error message
                  if suggestion generation fails.
    """
    try:
        # Set temperature to 0 for more consistent suggestions
        if hasattr(llm, "temperature"):
            llm.temperature = 0
        
        suggestion_generator_chain = await create_suggestion_generator_chain(llm)
        return await suggestion_generator_chain.invoke({
            "chat_history": history
        })
    except Exception as e:
        logger.error(f"Error in generate_suggestions: {str(e)}")
        return ["Unable to generate suggestions at this time"]
