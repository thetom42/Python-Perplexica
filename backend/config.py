"""
Configuration module for the Perplexica backend.

This module provides functions to load and retrieve configuration settings
from a TOML file, as well as update the configuration.
"""

import toml
from pathlib import Path
from typing import Dict, Any, Union, List

CONFIG_FILE = Path(__file__).parent.parent / "config.toml"

def load_config() -> Dict[str, Any]:
    """
    Load the configuration from the TOML file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration settings.
    """
    return toml.load(CONFIG_FILE)

def get_port() -> int:
    """
    Get the port number from the configuration.

    Returns:
        int: The port number.
    """
    config = load_config()
    try:
        return int(config["GENERAL"]["PORT"])
    except (KeyError, ValueError):
        return 8000 #default port

def get_similarity_measure() -> str:
    """
    Get the similarity measure from the configuration.

    Returns:
        str: The similarity measure.
    """
    config = load_config()
    try:
        return str(config["GENERAL"]["SIMILARITY_MEASURE"])
    except KeyError:
        return "cosine" #default

def get_openai_api_key() -> str:
    """
    Get the OpenAI API key from the configuration.

    Returns:
        str: The OpenAI API key.
    """
    config = load_config()
    try:
        return str(config["API_KEYS"]["OPENAI"])
    except KeyError:
        return "" #default

def get_groq_api_key() -> str:
    """
    Get the Groq API key from the configuration.

    Returns:
        str: The Groq API key.
    """
    config = load_config()
    try:
        return str(config["API_KEYS"]["GROQ"])
    except KeyError:
        return "" #default

def get_anthropic_api_key() -> str:
    """
    Get the Anthropic API key from the configuration.

    Returns:
        str: The Anthropic API key.
    """
    config = load_config()
    try:
        return str(config["API_KEYS"]["ANTHROPIC"])
    except KeyError:
        return "" #default

def get_searxng_api_endpoint() -> str:
    """
    Get the SearXNG API endpoint from the configuration.

    Returns:
        str: The SearXNG API endpoint.
    """
    config = load_config()
    try:
        return str(config["API_ENDPOINTS"]["SEARXNG"])
    except KeyError:
        return "" #default

def get_ollama_api_endpoint() -> str:
    """
    Get the Ollama API endpoint from the configuration.

    Returns:
        str: The Ollama API endpoint.
    """
    config = load_config()
    try:
        return str(config["API_ENDPOINTS"]["OLLAMA"])
    except KeyError:
        return "" #default

def get_wolfram_alpha_app_id() -> str:
    """
    Get the Wolfram Alpha App ID from the configuration.

    Returns:
        str: The Wolfram Alpha App ID, or an empty string if not set.
    """
    config = load_config()
    return str(config["API_KEYS"].get("WOLFRAM_ALPHA", ""))

def get_database_url() -> str:
    """
    Get the database URL from the configuration.

    Returns:
        str: The database URL, defaulting to a local SQLite database if not set.
    """
    config = load_config()
    return str(config.get("DATABASE", {}).get("URL", "sqlite:///./perplexica.db"))

def get_redis_url() -> str:
    """
    Get the Redis URL from the configuration.

    Returns:
        str: The Redis URL, defaulting to a local Redis instance if not set.
    """
    config = load_config()
    return str(config.get("CACHE", {}).get("REDIS_URL", "redis://localhost:6379/0"))

def get_log_level() -> str:
    """
    Get the logging level from the configuration.

    Returns:
        str: The logging level, defaulting to "INFO" if not set.
    """
    config = load_config()
    return str(config.get("LOGGING", {}).get("LEVEL", "INFO"))

def update_config(new_config: Dict[str, Any]) -> None:
    """
    Update the configuration file with new settings.

    Args:
        new_config (Dict[str, Any]): A dictionary containing the new configuration settings.
    """
    current_config = load_config()
    merged_config = {**current_config, **new_config}
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        toml.dump(merged_config, f)

def get_chat_model_config() -> Dict[str, Any]:
    """
    Get the chat model configuration.

    Returns:
        Dict[str, Any]: A dictionary containing the chat model configuration.
    """
    config = load_config()
    return config.get("CHAT_MODEL", {}) or {}

def get_embedding_model_config() -> Dict[str, Any]:
    """
    Get the embedding model configuration.

    Returns:
        Dict[str, Any]: A dictionary containing the embedding model configuration.
    """
    config = load_config()
    return config.get("EMBEDDING_MODEL", {}) or {}

def get_cors_origins() -> List[str]:
    """
    Get the CORS origins from the configuration.

    Returns:
        list: A list of allowed CORS origins, defaulting to ["*"] if not set.
    """
    config = load_config()
    cors_origins = config.get("SECURITY", {}).get("CORS_ORIGINS", ["*"])
    return [str(origin) for origin in cors_origins]

def get_api_keys() -> Dict[str, Union[str, None]]:
    config = load_config()
    return config.get("API_KEYS", {}) or {}

def get_api_endpoints() -> Dict[str, str]:
    config = load_config()
    return config.get("API_ENDPOINTS", {}) or {}

def get_database_config() -> Dict[str, str]:
    config = load_config()
    return config.get("DATABASE", {}) or {}

def get_cache_config() -> Dict[str, str]:
    config = load_config()
    return config.get("CACHE", {}) or {}

def get_logging_config() -> Dict[str, str]:
    config = load_config()
    return config.get("LOGGING", {}) or {}

def get_security_config() -> Dict[str, List[str]]:
    config = load_config()
    return config.get("SECURITY", {}) or {}

def get_chat_model() -> Dict[str, str]:
    config = load_config()
    return config.get("CHAT_MODEL", {}) or {}

def get_embedding_model() -> Dict[str, str]:
    config = load_config()
    return config.get("EMBEDDING_MODEL", {}) or {}

def get_general_config() -> Dict[str, Union[str, int]]:
    config = load_config()
    return config.get("GENERAL", {}) or {}
