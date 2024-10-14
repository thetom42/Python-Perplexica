"""
Line Output Parser Module

This module provides a custom output parser that extracts a single line of text
from a structured output, removing any leading list markers or bullet points.
"""

from langchain.schema import BaseOutputParser
import re
from typing import ClassVar

class LineOutputParser(BaseOutputParser[str]):
    """
    A custom output parser that extracts a single line of text from a structured output.

    This parser looks for content between specified XML-like tags and removes any
    leading list markers or bullet points from the extracted text.

    Attributes:
        key (str): The XML-like tag name to look for in the input text.
    """

    key: ClassVar[str] = "questions"

    def parse(self, text: str) -> str:
        """
        Parse the input text and extract a single line of content.

        This method searches for content between <{key}> and </{key}> tags,
        removes any leading list markers or bullet points, and returns the cleaned text.

        Args:
            text (str): The input text to parse.

        Returns:
            str: The extracted and cleaned line of text, or an empty string if the tags are not found.
        """
        regex = r"^(\s*(-|\*|\d+\.\s|\d+\)\s|\u2022)\s*)+"
        start_key_index = text.find(f"<{self.key}>")
        end_key_index = text.find(f"</{self.key}>")

        if start_key_index == -1 or end_key_index == -1:
            return ""

        questions_start_index = start_key_index + len(f"<{self.key}>")
        questions_end_index = end_key_index
        line = text[questions_start_index:questions_end_index].strip()
        line = re.sub(regex, "", line)

        return line

    @property
    def _type(self) -> str:
        """
        Get the type identifier for this output parser.

        Returns:
            str: The type identifier string.
        """
        return "line_output_parser"

    def get_format_instructions(self) -> str:
        """
        Get the format instructions for this output parser.

        This method is not implemented for this parser.

        Raises:
            NotImplementedError: This method is not implemented for this parser.
        """
        raise NotImplementedError("Not implemented for this output parser.")
