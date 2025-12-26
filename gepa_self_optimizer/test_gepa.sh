#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Resolve Project Root ---
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Get the project root directory (one level up from the script's directory)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# --- Virtual Environment Configuration ---
VENV_DIR="$HOME/virtualenvs/dspy_env" # Assuming 'dspy_env' is the name. Change if needed.
VENV_BIN="$VENV_DIR/bin"

if [ ! -d "$VENV_DIR" ]; then
    echo "⚠️  Virtual environment not found at $VENV_DIR."
    echo "  Creating a new virtual environment here..."
    mkdir -p "$(dirname "$VENV_DIR")"
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to create virtual environment at $VENV_DIR"
        exit 1
    fi
fi

echo "🔧 Activating virtual environment at $VENV_DIR"
source "$VENV_DIR/bin/activate"

# --- Configuration ---
# Path to the requirements file, relative to the project root
REQUIREMENTS_FILE="$PROJECT_ROOT/gepa_self_optimizer/requirements.txt"

echo "=================================================================="
echo "GEPA Self Optimizer Test Runner"
echo "=================================================================="

# --- Dependency Installation ---
if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "❌ Error: requirements.txt not found at $REQUIREMENTS_FILE"
  echo "  This file is required to install project dependencies (e.g., pytest, dspy)."
  exit 1
fi

echo "📥 Installing/upgrading dependencies from $REQUIREMENTS_FILE..."
# Upgrade pip and install requirements within the activated venv
pip install --upgrade pip
pip install -r "$REQUIREMENTS_FILE"

# --- Run Tests ---
echo
echo "🧪 Running tests with pytest..."
echo "------------------------------------------------------------------"
# Run pytest on the specific test file. The 'set -e' will stop the script if tests fail.
# Discover and run all tests within the specified module/file.
python -m pytest "$PROJECT_ROOT/gepa_self_optimizer/test_gepa.py" -v
echo "------------------------------------------------------------------"

echo
echo "🏁 All steps completed successfully."
