# Python-Perplexica

## This repository is a fork of the [original Perplexica project](https://github.com/ItzCrazyKns/Perplexica), demonstrating the transformation of its TypeScript backend into a Python backend. This transformation was achieved using the Visual Studio Code extension "cline" (formerly "claude-dev") and the Claude Sonnet 3.5 language model.

The process involved two main stages:

1. **Analysis of the Source Code:**  An initial analysis of the original project's `src` directory was performed using Claude Sonnet 3.5. This analysis informed the subsequent transformation process.  The chat log detailing this analysis can be found in `cline-explain-structure-of-src-dir.md`.

2. **Code Transformation:** Claude Sonnet 3.5 was then used to transform the TypeScript codebase into Python. The transformed code is stored in the `backend` directory.  The chat log documenting this transformation is available in `cline-task-transform-backend-into-python.md`.  The original TypeScript code can still be found in the `src` directory.

The transformation process, as documented in `cline-task-transform-backend-into-python.md`, involved significant interaction with the LLM.  While the exact cost is not explicitly stated, the process was resource-intensive, frequently reaching both per-minute and daily token limits for Claude Sonnet 3.5.  These limitations necessitated frequent pauses, extending the overall project timeline.  Despite these interruptions, the LLM proved effective in translating the complex TypeScript codebase into a functional Python equivalent.

This fork serves as a practical example of using "cline" and large language models for codebase refactoring and language migration.  It showcases the capabilities of these tools in a real-world scenario.
