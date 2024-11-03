"""
Enums used throughout the application.

This module contains all the enumerated types used for validation and type safety
across the application. These enums help ensure consistent values are used for
modes, types, and other categorical data.
"""

from enum import Enum


class OptimizationMode(str, Enum):
    """
    Optimization modes for request processing.
    
    These modes determine the balance between speed and quality of results:
    - BALANCED: Default mode, balancing speed and quality
    - SPEED: Prioritize faster response times
    - QUALITY: Prioritize result quality
    """
    BALANCED = "balanced"
    SPEED = "speed"
    QUALITY = "quality"


class FocusMode(str, Enum):
    """
    Focus modes for search operations.
    
    These modes determine the type of search to perform:
    - WEB: General web search
    - ACADEMIC: Academic/research focused search
    - IMAGE: Image search
    - VIDEO: Video search
    - WRITING: Writing assistance
    """
    WEB = "web"
    ACADEMIC = "academic"
    IMAGE = "image"
    VIDEO = "video"
    WRITING = "writing"


class Role(str, Enum):
    """
    Role types for chat messages.
    
    These roles identify the sender of a message:
    - USER: Message from the user
    - ASSISTANT: Message from the AI assistant
    """
    USER = "user"
    ASSISTANT = "assistant"
