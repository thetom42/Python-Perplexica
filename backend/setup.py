"""Setup script for the backend.

This script:
    1. Creates a virtual environment
    2. Installs required packages
    3. Creates a default .env file if it doesn't exist
    4. Runs database migrations
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main setup function."""
    # Create .env if it doesn't exist
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if not os.path.exists(env_path):
        sample_env = {
            "PORT": "3001",
            "SIMILARITY_MEASURE": "cosine",
            "OPENAI_API_KEY": "",
            "GROQ_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "WOLFRAM_ALPHA_APP_ID": "",
            "SEARXNG_API_ENDPOINT": "http://localhost:32768",
            "OLLAMA_API_ENDPOINT": "",
            "DATABASE_URL": "sqlite:///./perplexica.db",
            "REDIS_URL": "redis://localhost:6379/0",
            "LOG_LEVEL": "INFO",
            "CORS_ORIGINS": '["*"]',
            "CHAT_MODEL_PROVIDER": "openai",
            "CHAT_MODEL_NAME": "gpt-3.5-turbo",
            "EMBEDDING_MODEL_PROVIDER": "openai",
            "EMBEDDING_MODEL_NAME": "text-embedding-ada-002"
        }

        with open(env_path, "w", encoding="utf-8") as f:
            for key, value in sample_env.items():
                f.write(f"{key}={value}\n")

        print("Created .env with default values. Please update it with your actual API keys and endpoints.")

    # Install requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    # Run database migrations
    subprocess.check_call([sys.executable, "-m", "alembic", "upgrade", "head"])

if __name__ == "__main__":
    main()
