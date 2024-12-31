from dataclasses import dataclass
from typing import Optional

@dataclass
class BaseLLMRequest:
    prompt: str
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

@dataclass
class SearchRequest:
    query: str
    focus_mode: str
    optimization_mode: str
    prompt: str
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

@dataclass
class VideoSearchRequest:
    query: str
    focus_mode: str
    optimization_mode: str
    prompt: str
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
