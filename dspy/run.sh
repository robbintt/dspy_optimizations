#!/bin/bash
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <script_name>"
    echo "Example: ./run.sh optimize_prompt.py"
    exit 1
fi

SCRIPT_NAME=$1

# Check if script exists
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: Script '$SCRIPT_NAME' not found."
    exit 1
fi

# Activate virtual environment
VENV_PATH="$HOME/virtualenvs/dspy_venv"
if [ -d "$VENV_PATH" ]; then
    echo "--- Activating virtual environment ---"
    source "$VENV_PATH/bin/activate"
else
    echo "Error: Virtual environment 'dspy_venv' not found. Please run ./setup.sh first."
    exit 1
fi

# Load environment variables
if [ -f ".env" ]; then
    echo "--- Loading environment variables from .env ---"
    set -a
    source .env
    set +a
else
    echo "Warning: .env file not found. Script might fail if it requires environment variables."
fi

# Run the Python script using the extracted Python runner
python scripts/run_script.py "$SCRIPT_NAME"
