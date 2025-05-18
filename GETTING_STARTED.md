# AURA: Getting Started Guide

AURA (Augmented Unsupervised Research Analyzer) is a powerful tool for research paper analysis that helps extract keywords, retrieve articles, cluster abstracts, and generate summaries. This guide will help you set up and use AURA either with VS Code or Claude Desktop.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Environment Setup](#environment-setup)
4. [Using AURA with VS Code](#using-aura-with-vs-code)
5. [Using AURA with Claude Desktop](#using-aura-with-claude-desktop)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.11 or higher
- pip or uv (package manager)
- Git (for cloning the repository)
- Google Gemini API key (for keyword extraction and detailed summaries)
- VS Code or Claude Desktop

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/AINE2425/AURA.git
   cd AURA
   ```

2. **Set up a virtual environment**:

   Using Python's built-in venv:

   ```bash
   cd backend
   python -m venv .venv
   ```

   Using uv (faster virtual environment creation):

   ```bash
   cd backend
   pip install uv
   uv venv
   ```

3. **Activate the virtual environment**:

   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**:

   Using pip:

   ```bash
   pip install -e .
   ```

   Using uv (faster installation):

   ```bash
   pip install uv # if not already installed
   uv pip install -e .
   ```

## Environment Setup

1. **Create a `.env` file in the `backend` directory with the following content**:

   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   MODEL_MAX_TOKENS=1000000
   ```

2. **Get a Gemini API key**:
   - Visit [Google AI Studio](https://aistudio.google.com/)
   - Create an account or sign in
   - Go to API section to create an API key
   - Place this key in your `.env` file

## Using AURA with VS Code

### Setup VS Code

1. **Install VS Code extensions**:

   - Python Extension for VS Code
   - Jupyter Extension (optional, for notebook support)

2. **Open the project in VS Code**:

   ```bash
   code .
   ```

3. **Select the Python interpreter**:
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
   - Type "Python: Select Interpreter"
   - Choose the interpreter from the `.venv` folder you created earlier

### Running AURA in VS Code

1. **Configure VS Code for MCP**:

   - Open VS Code settings.json (press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS) and type "Open Settings (JSON)")
   - Add the following configuration (modify paths to match your installation):

   ```json
   "mcp": {
       "servers": {
           "AURA - Augmented Unsupervised Research Agent": {
               "type": "stdio",
               "command": "/path/to/AURA/backend/.venv/bin/python",
               "args": [
                   "/path/to/AURA/backend/src/server.py"
               ]
           }
       }
   }
   ```

2. **Start the AURA server**:

   - In VS Code, open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`)
   - Type "MCP: Connect to Server" and select it
   - Choose "AURA - Augmented Unsupervised Research Agent" from the menu

3. **Interact with AURA**:
   - Once connected, you can send requests to AURA through the VS Code chat interface (Make sure Agent mode is selected)
   - Use the available tools like keyword extraction, clustering, and summarization
   - In the chat window, you can ask questions like:
     - "Extract keywords from this abstract: [paste abstract]"
     - "Search for papers related to these keywords: [keywords]"
     - "Cluster these abstracts and visualize the results: [paste abstracts]"

## Using AURA with Claude Desktop

Claude Desktop makes it easy to interact with AURA's capabilities through natural language. Here's how to set it up:

1. **Download and Install Claude Desktop** from https://claude.ai/desktop

2. **Install MCP server**:

- Use `fastmcp` CLI for easy installing on Claude Desktop:

```bash
fastmcp install src/server.py
```

3. **Prompt Claude to help you interact with AURA**:

   Example prompts:

   ```
   "Using AURA,- "Extract keywords from this abstract: [paste abstract]"
   ```

- "Search for papers related to these keywords: [keywords]"
- "Cluster these abstracts and visualize the results: [paste abstracts]"

## Troubleshutting

If you encounter issues not covered here, please check:

- The project README.md file for additional information
- Open an issue on the project's GitHub repository

Happy researching with AURA!
