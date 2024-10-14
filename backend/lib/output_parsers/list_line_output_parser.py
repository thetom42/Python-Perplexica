from langchain.schema import BaseOutputParser
from typing import List
import re

class ListLineOutputParser(BaseOutputParser[List[str]]):
    key: str = "questions"

    def parse(self, text: str) -> List[str]:
        regex = r"^(\s*(-|\*|\d+\.\s|\d+\)\s|\u2022)\s*)+"
        start_key_index = text.find(f"<{self.key}>")
        end_key_index = text.find(f"</{self.key}>")

        if start_key_index == -1 or end_key_index == -1:
            return []

        questions_start_index = start_key_index + len(f"<{self.key}>")
        questions_end_index = end_key_index
        lines = text[questions_start_index:questions_end_index].strip().split("\n")
        lines = [line.strip() for line in lines if line.strip()]
        lines = [re.sub(regex, "", line) for line in lines]

        return lines

    @property
    def _type(self) -> str:
        return "list_line_output_parser"

    def get_format_instructions(self) -> str:
        raise NotImplementedError("Not implemented for this output parser.")
