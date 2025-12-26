#!/bin/bash
set -e

# --- Configuration ---
PROJECT_NAME="microagent_helloworld"
VENV_PATH="$HOME/virtualenvs/$PROJECT_NAME"

# --- Script Logic ---
echo "--- MicroAgent HelloWorld Demo Runner ---"

# Get the directory the script is located in
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script's directory so `uv` can find pyproject.toml
cd "$SCRIPT_DIR"

# 1. Check for LLM API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: The 'OPENAI_API_KEY' environment variable is not set."
    echo "   Please set it to run the demo:"
    echo "   export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# 2. Create Virtual Environment
echo "🔧 Ensuring virtual environment at $VENV_PATH..."
mkdir -p "$(dirname "$VENV_PATH")"
if [ ! -d "$VENV_PATH" ]; then
    uv venv "$VENV_PATH"
else
    echo "✅ Virtual environment already exists."
fi

# 3. Install Dependencies
echo "📦 Installing dependencies..."
UV_PROJECT_VIRTUAL_ENV="$VENV_PATH" uv sync

# 4. Run the Demo
echo "🚀 Running demo..."
echo "---"
UV_PROJECT_VIRTUAL_ENV="$VENV_PATH" uv run helloworld-demo
echo "---"

echo "✅ Demo complete!"
