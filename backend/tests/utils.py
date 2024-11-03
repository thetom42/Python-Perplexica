from typing import Dict, Any, Optional
from routes.shared.base import OptimizationMode, ChatModel, EmbeddingModel


def create_test_chat_model(
    provider: str = "test_provider",
    model: str = "test_model",
    custom_url: Optional[str] = None,
    custom_key: Optional[str] = None
) -> Dict[str, Any]:
    """Create a test chat model configuration."""
    model_config = {
        "provider": provider,
        "model": model,
    }
    if custom_url:
        model_config["customOpenAIBaseURL"] = custom_url
    if custom_key:
        model_config["customOpenAIKey"] = custom_key
    return model_config


def create_test_embedding_model(
    provider: str = "test_provider",
    model: str = "test_model"
) -> Dict[str, Any]:
    """Create a test embedding model configuration."""
    return {
        "provider": provider,
        "model": model,
    }


def create_base_request(
    chat_model: Optional[Dict[str, Any]] = None,
    embedding_model: Optional[Dict[str, Any]] = None,
    optimization_mode: str = OptimizationMode.BALANCED
) -> Dict[str, Any]:
    """Create a base request with optional model configurations."""
    request: Dict[str, Any] = {
        "optimization_mode": optimization_mode,
    }
    if chat_model:
        request["chat_model"] = chat_model
    if embedding_model:
        request["embedding_model"] = embedding_model
    return request
