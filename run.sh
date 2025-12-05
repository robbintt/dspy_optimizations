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
if [ -d "venv" ]; then
    echo "--- Activating virtual environment ---"
    source venv/bin/activate
else
    echo "Error: Virtual environment 'venv' not found. Please run ./setup.sh first."
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

echo ""
echo "--- Running python $SCRIPT_NAME ---"
python "$SCRIPT_NAME"

echo ""
echo "--- Script finished ---"
