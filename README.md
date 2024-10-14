# Python-Perplexica

## This repository is a fork of the [original Perplexica project](https://github.com/ItzCrazyKns/Perplexica), demonstrating the transformation of its TypeScript backend into a Python backend. This transformation was achieved using the Visual Studio Code extension "cline" (formerly "claude-dev") and the Claude Sonnet 3.5 language model.

The process involved two main stages:

1. **Analysis of the Source Code:**  An initial analysis of the original project's `src` directory was performed using Claude Sonnet 3.5. This analysis informed the subsequent transformation process.  The chat log detailing this analysis can be found in `cline-explain-structure-of-src-dir.md`.

2. **Code Transformation:** Claude Sonnet 3.5 was then used to transform the TypeScript codebase into Python. The transformed code is stored in the `backend` directory.  The chat log documenting this transformation is available in `cline-task-transform-backend-into-python.md`.  The original TypeScript code can still be found in the `src` directory.

The transformation process, as documented in `cline-task-transform-backend-into-python.md`, involved significant interaction with the LLM.  While the exact cost is not explicitly stated, the process was resource-intensive, frequently reaching both per-minute and daily token limits for Claude Sonnet 3.5.  These limitations necessitated frequent pauses, extending the overall project timeline.  Despite these interruptions, the LLM proved effective in translating the complex TypeScript codebase into a functional Python equivalent.

This fork serves as a practical example of using "cline" and large language models for codebase refactoring and language migration.  It showcases the capabilities of these tools in a real-world scenario.

## Development with Dev Container

This project now supports development using Visual Studio Code's Dev Containers. This feature allows you to use a Docker container as a full-featured development environment, ensuring consistency across different development machines.

### Prerequisites

1. Install [Docker](https://www.docker.com/get-started) on your machine.
2. Install [Visual Studio Code](https://code.visualstudio.com/).
3. Install the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension in VS Code.

### Using the Dev Container

1. Clone this repository to your local machine.
2. Open the project folder in Visual Studio Code.
3. When prompted, click "Reopen in Container" or run the "Remote-Containers: Reopen in Container" command from the Command Palette (F1).
4. VS Code will build the Docker image and start the container. This may take a few minutes the first time.
5. Once the container is running, the post-create command will automatically install all Python and Node.js dependencies.
6. You'll now have a fully configured development environment with all necessary dependencies installed.

The Dev Container includes:
- Python 3.11
- Node.js 18 (installed via devcontainer features)
- Yarn
- All Python dependencies specified in `backend/requirements.txt`
- All Node.js dependencies specified in `package.json`
- VS Code extensions for Python, Pylance, ESLint, and Prettier

You can now develop, run, and debug your code directly within the container. The backend directory is added to the PATH, allowing easier execution of Python scripts.

### Customization

If you need to customize the Dev Container configuration, you can modify the following files:
- `.devcontainer/devcontainer.json`: Configures the Dev Container features, VS Code settings, and post-create commands.
- `.devcontainer/Dockerfile`: Defines the base Docker image and additional setup steps.

Key aspects of the current setup:
- The Dockerfile uses a Python 3.11 base image and sets up the working environment.
- Node.js is installed via devcontainer features in the devcontainer.json file.
- Dependencies are installed via a post-create command, ensuring they're always up-to-date.

Remember to rebuild the container after making changes to these files by running the "Remote-Containers: Rebuild Container" command from the Command Palette.
