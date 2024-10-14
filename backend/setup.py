"""
Setup script for the Perplexica backend.

This module provides functions to set up the development environment,
including creating a virtual environment, installing dependencies,
and initializing the configuration file.
"""

import os
import subprocess
import toml

def run_command(command: str) -> str:
    """
    Execute a shell command and return its output.

    Args:
        command (str): The command to execute.

    Returns:
        str: The command's stdout output.

    Raises:
        SystemExit: If the command execution fails.
    """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(stderr)
        exit(1)
    return stdout

def setup_environment() -> None:
    """
    Set up the development environment for the Perplexica backend.

    This function performs the following tasks:
    1. Creates a virtual environment
    2. Installs required packages
    3. Creates a default config.toml file if it doesn't exist
    4. Runs database migrations

    Raises:
        SystemExit: If any step in the setup process fails.
    """
    print("Setting up the development environment...")

    # Get virtual environment name from environment variable or use default
    venv_name = os.environ.get("VIRTUAL_ENV_NAME", "venv")

    # Create virtual environment
    run_command(f"python3.12 -m venv {venv_name}")

    # Activate virtual environment
    if os.name == 'nt':  # Windows
        activate_cmd = f".\\{venv_name}\\Scripts\\activate"
    else:  # macOS and Linux
        activate_cmd = f"source {venv_name}/bin/activate"

    # Install requirements
    run_command(f"{activate_cmd} && pip install -r requirements.txt")

    # Create config.toml if it doesn't exist
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.toml')
    if not os.path.exists(config_path):
        config = {
            "GENERAL": {
                "PORT": 8000,
                "SIMILARITY_MEASURE": "cosine"
            },
            "API_KEYS": {
                "OPENAI": "your_openai_api_key",
                "GROQ": "your_groq_api_key",
                "ANTHROPIC": "your_anthropic_api_key",
                "WOLFRAM_ALPHA": "your_wolfram_alpha_app_id"
            },
            "API_ENDPOINTS": {
                "SEARXNG": "http://your_searxng_instance_url",
                "OLLAMA": "http://your_ollama_instance_url"
            },
            "DATABASE": {
                "URL": "sqlite:///./perplexica.db"
            },
            "CACHE": {
                "REDIS_URL": "redis://localhost:6379/0"
            },
            "LOGGING": {
                "LEVEL": "INFO"
            },
            "SECURITY": {
                "CORS_ORIGINS": ["http://localhost:3000"]
            },
            "CHAT_MODEL": {
                "PROVIDER": "openai",
                "MODEL": "gpt-3.5-turbo"
            },
            "EMBEDDING_MODEL": {
                "PROVIDER": "openai",
                "MODEL": "text-embedding-ada-002"
            }
        }
        with open(config_path, "w", encoding="UTF-8") as f:
            toml.dump(config, f)
        print("Created config.toml with default values. Please update it with your actual API keys and endpoints.")

    # Run database migrations
    run_command(f"{activate_cmd} && alembic upgrade head")

    print("Development environment setup complete!")
    print("Please ensure Redis is installed and running on your system.")
    print("You can now start the backend server using: uvicorn main:app --reload")

if __name__ == "__main__":
    setup_environment()
