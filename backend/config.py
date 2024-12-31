"""Configuration module for loading and managing application settings."""

import os
from pathlib import Path
from typing import TypedDict, Any, cast, Literal, NotRequired, Dict
import tomli
import tomli_w


CONFIG_FILENAME = 'config.toml'


class GeneralConfig(TypedDict):
    """Type definition for general configuration settings."""
    PORT: int  # Port to run the server on
    SIMILARITY_MEASURE: str  # "cosine" or "dot"
    CORS_ORIGINS: list[str]  # List of allowed CORS origins


class ApiKeysConfig(TypedDict):
    """Type definition for API key configuration settings."""
    OPENAI: str  # OpenAI API key
    GROQ: str  # Groq API key
    ANTHROPIC: str  # Anthropic API key


class ApiEndpointsConfig(TypedDict):
    """Type definition for API endpoint configuration settings."""
    SEARXNG: str  # SearxNG API URL
    OLLAMA: str  # Ollama API URL


class Config(TypedDict):
    """Type definition for the complete configuration."""
    GENERAL: GeneralConfig
    API_KEYS: ApiKeysConfig
    API_ENDPOINTS: ApiEndpointsConfig


# Type for configuration section keys
ConfigKey = Literal['GENERAL', 'API_KEYS', 'API_ENDPOINTS']


# Partial versions of the config types for updates
class PartialGeneralConfig(TypedDict, total=False):
    """Partial type definition for general configuration settings."""
    PORT: NotRequired[int]
    SIMILARITY_MEASURE: NotRequired[str]
    CORS_ORIGINS: NotRequired[list[str]]


class PartialApiKeysConfig(TypedDict, total=False):
    """Partial type definition for API key configuration settings."""
    OPENAI: NotRequired[str]
    GROQ: NotRequired[str]
    ANTHROPIC: NotRequired[str]


class PartialApiEndpointsConfig(TypedDict, total=False):
    """Partial type definition for API endpoint configuration settings."""
    SEARXNG: NotRequired[str]
    OLLAMA: NotRequired[str]


class PartialConfig(TypedDict, total=False):
    """Partial type definition for the complete configuration."""
    GENERAL: NotRequired[PartialGeneralConfig]
    API_KEYS: NotRequired[PartialApiKeysConfig]
    API_ENDPOINTS: NotRequired[PartialApiEndpointsConfig]


def _get_config_path() -> Path:
    """Get the path to the config file."""
    return Path(__file__).parent.parent / CONFIG_FILENAME


def load_config() -> Config:
    """Load the configuration from the TOML file."""
    try:
        with open(_get_config_path(), 'rb') as f:
            return cast(Config, tomli.load(f))
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {str(e)}") from e


def get_port() -> int:
    """Get the port number from configuration."""
    return load_config()['GENERAL']['PORT']


def get_similarity_measure() -> str:
    """Get the similarity measure from configuration."""
    return load_config()['GENERAL']['SIMILARITY_MEASURE']


def get_openai_api_key() -> str:
    """Get the OpenAI API key from configuration."""
    return load_config()['API_KEYS']['OPENAI']


def get_groq_api_key() -> str:
    """Get the Groq API key from configuration."""
    return load_config()['API_KEYS']['GROQ']


def get_anthropic_api_key() -> str:
    """Get the Anthropic API key from configuration."""
    return load_config()['API_KEYS']['ANTHROPIC']


def get_searxng_api_endpoint() -> str:
    """Get the SearXNG API endpoint from configuration or environment variable."""
    return os.getenv('SEARXNG_API_URL') or load_config()['API_ENDPOINTS']['SEARXNG']


def get_ollama_api_endpoint() -> str:
    """Get the Ollama API endpoint from configuration."""
    return load_config()['API_ENDPOINTS']['OLLAMA']


def get_cors_origins() -> list[str]:
    """Get the list of allowed CORS origins from configuration."""
    return load_config()['GENERAL']['CORS_ORIGINS']


def update_config(new_config: PartialConfig) -> None:
    """
    Update the configuration file with new values while preserving existing ones.
    
    Args:
        new_config: Dictionary containing the new configuration values.
    """
    current_config = load_config()
    sections: list[ConfigKey] = ['GENERAL', 'API_KEYS', 'API_ENDPOINTS']

    # Create a new dictionary that will be written to TOML
    toml_config: Dict[str, Dict[str, Any]] = {}

    # Copy current config and merge with new values
    for section in sections:
        toml_config[section] = dict(current_config[section])
        if section in new_config and isinstance(new_config.get(section), dict):
            new_section = new_config.get(section, {})
            if new_section:
                for key, value in new_section.items():
                    if value is not None:
                        toml_config[section][key] = value

    # Write the updated configuration
    try:
        with open(_get_config_path(), 'wb') as f:
            tomli_w.dump(toml_config, f)
    except Exception as e:
        raise RuntimeError(f"Failed to update config: {str(e)}") from e
