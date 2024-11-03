"""
Configuration module for the Perplexica backend.

This module provides functions to load and retrieve configuration settings
from environment variables.
"""

import json
import os
from typing import Dict, Any, Union, List, cast
from pathlib import Path
from dotenv import load_dotenv
from pydantic.types import SecretStr
from langchain_core.utils.utils import convert_to_secret_str

# Load environment variables from .env file
load_dotenv()

def get_port() -> int:
    """
    Get the port number from the configuration.

    Returns:
        int: The port number.
    """
    try:
        return int(os.getenv("PORT", "8000"))
    except ValueError:
        return 8000  # default port

def get_similarity_measure() -> str:
    """
    Get the similarity measure from the configuration.

    Returns:
        str: The similarity measure.
    """
    return os.getenv("SIMILARITY_MEASURE", "cosine")

def get_openai_api_key() -> SecretStr:
    """
    Get the OpenAI API key from the configuration.

    Returns:
        str: The OpenAI API key.
    """
    return convert_to_secret_str(os.getenv("OPENAI_API_KEY", ""))

def get_groq_api_key() -> SecretStr:
    """
    Get the Groq API key from the configuration.

    Returns:
        str: The Groq API key.
    """
    return convert_to_secret_str(os.getenv("GROQ_API_KEY", ""))

def get_anthropic_api_key() -> SecretStr:
    """
    Get the Anthropic API key from the configuration.

    Returns:
        str: The Anthropic API key.
    """
    return convert_to_secret_str(os.getenv("ANTHROPIC_API_KEY", ""))

def get_searxng_api_endpoint() -> str:
    """
    Get the SearXNG API endpoint from the configuration.

    Returns:
        str: The SearXNG API endpoint.
    """
    return os.getenv("SEARXNG_API_ENDPOINT", "")

def get_ollama_api_endpoint() -> str:
    """
    Get the Ollama API endpoint from the configuration.

    Returns:
        str: The Ollama API endpoint.
    """
    return os.getenv("OLLAMA_API_ENDPOINT", "")

def get_wolfram_alpha_app_id() -> str:
    """
    Get the Wolfram Alpha App ID from the configuration.

    Returns:
        str: The Wolfram Alpha App ID, or an empty string if not set.
    """
    return os.getenv("WOLFRAM_ALPHA_APP_ID", "")

def get_database_url() -> str:
    """
    Get the database URL from the configuration.

    Returns:
        str: The database URL, defaulting to a local SQLite database if not set.
    """
    return os.getenv("DATABASE_URL", "sqlite:///./perplexica.db")

def get_redis_url() -> str:
    """
    Get the Redis URL from the configuration.

    Returns:
        str: The Redis URL, defaulting to a local Redis instance if not set.
    """
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")

def get_log_level() -> str:
    """
    Get the logging level from the configuration.

    Returns:
        str: The logging level, defaulting to "INFO" if not set.
    """
    return os.getenv("LOG_LEVEL", "INFO")

def get_cors_origins() -> List[str]:
    """
    Get the CORS origins from the configuration.

    Returns:
        list: A list of allowed CORS origins, defaulting to ["*"] if not set.
    """
    cors_origins = os.getenv("CORS_ORIGINS", '["*"]')
    try:
        origins: List[str] = json.loads(cors_origins)
        return origins
    except json.JSONDecodeError:
        return ["*"]

def get_api_keys() -> Dict[str, Any]:
    """
    Get all API keys from the configuration.

    Returns:
        Dict[str, str]: A dictionary containing all API keys.
    """
    return {
        "OPENAI": get_openai_api_key(),
        "GROQ": get_groq_api_key(),
        "ANTHROPIC": get_anthropic_api_key(),
        "WOLFRAM_ALPHA": get_wolfram_alpha_app_id()
    }

def get_api_endpoints() -> Dict[str, str]:
    """
    Get all API endpoints from the configuration.

    Returns:
        Dict[str, str]: A dictionary containing all API endpoints.
    """
    return {
        "SEARXNG": get_searxng_api_endpoint(),
        "OLLAMA": get_ollama_api_endpoint()
    }

def get_database_config() -> Dict[str, str]:
    """
    Get the database configuration.

    Returns:
        Dict[str, str]: A dictionary containing the database configuration.
    """
    return {"URL": get_database_url()}

def get_cache_config() -> Dict[str, str]:
    """
    Get the cache configuration.

    Returns:
        Dict[str, str]: A dictionary containing the cache configuration.
    """
    return {"REDIS_URL": get_redis_url()}

def get_logging_config() -> Dict[str, str]:
    """
    Get the logging configuration.

    Returns:
        Dict[str, str]: A dictionary containing the logging configuration.
    """
    return {"LEVEL": get_log_level()}

def get_security_config() -> Dict[str, List[str]]:
    """
    Get the security configuration.

    Returns:
        Dict[str, List[str]]: A dictionary containing the security configuration.
    """
    return {"CORS_ORIGINS": get_cors_origins()}

def get_chat_model() -> Dict[str, str]:
    """
    Get the chat model configuration.

    Returns:
        Dict[str, str]: A dictionary containing the chat model configuration.
    """
    return {
        "PROVIDER": os.getenv("CHAT_MODEL_PROVIDER", "openai"),
        "NAME": os.getenv("CHAT_MODEL_NAME", "gpt-3.5-turbo")
    }

def get_embedding_model() -> Dict[str, str]:
    """
    Get the embedding model configuration.

    Returns:
        Dict[str, str]: A dictionary containing the embedding model configuration.
    """
    return {
        "PROVIDER": os.getenv("EMBEDDING_MODEL_PROVIDER", "openai"),
        "NAME": os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
    }

def get_general_config() -> Dict[str, Union[str, int]]:
    """
    Get the general configuration.

    Returns:
        Dict[str, Union[str, int]]: A dictionary containing the general configuration.
    """
    return {
        "PORT": get_port(),
        "SIMILARITY_MEASURE": get_similarity_measure()
    }
