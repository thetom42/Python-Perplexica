from dataclasses import dataclass
from typing import List, Optional

@dataclass
class StreamResponse:
    content: str
    is_final: bool = False

@dataclass
class SuggestionResponse:
    suggestions: List[str]
    context: Optional[str] = None

@dataclass
class VideoResult:
    title: str
    url: str
    duration: str
    thumbnail: str

@dataclass
class VideoSearchResponse:
    results: List[VideoResult]
    query: str
    optimization_mode: str
