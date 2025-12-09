#!/bin/bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <script_name>"
    echo "Example: ./run.sh optimize_prompt.py"
    exit 1
fi

SCRIPT_NAME=$1

# Check if script exists in the dspy directory
if [ ! -f "$SCRIPT_DIR/$SCRIPT_NAME" ]; then
    echo "Error: Script '$SCRIPT_NAME' not found in $SCRIPT_DIR"
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
elif [ -f "../.env" ]; then
    echo "--- Loading environment variables from root .env ---"
    set -a
    source ../.env
    set +a
else
    echo "Warning: .env file not found. Script might fail if it requires environment variables."
fi

# Change to the script directory and run the Python script
cd "$SCRIPT_DIR"
python "$SCRIPT_NAME"
