#!/bin/bash

# MDAP Harness Run Script
# Provides convenient commands to run MDAP examples and tests

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$HOME/virtualenvs/mdap_harness_venv"

# Function to check if virtual environment exists
check_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        echo -e "\033[0;31m[ERROR]\033[0m Virtual environment not found!"
        echo "Please run './setup_mdap.sh' first to set up the environment."
        exit 1
    fi
}

# Function to activate virtual environment
activate_venv() {
    echo -e "\033[0;34m[MDAP]\033[0m Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
}

# Function to check if .env file exists (optional)
check_env() {
    # Find parent directory with .env file
    PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    ENV_FILE="$PARENT_DIR/.env"
    
    if [ -f "$ENV_FILE" ]; then
        echo -e "\033[0;34m[MDAP]\033[0m Found .env file at $ENV_FILE"
        # Export .env file path for Python scripts
        export ENV_FILE_PATH="$ENV_FILE"
    else
        echo -e "\033[1;33m[MDAP]\033[0m No .env file found at $ENV_FILE, using system defaults"
        echo "You can optionally create one from .env.example:"
        echo "  cp .env.example .env"
        echo "Then edit .env with your API keys."
    fi
}

# Main script logic
main() {
    # Check prerequisites
    check_venv
    activate_venv
    check_env
    
    # Check if we're running an example
    if [ "$1" = "example" ]; then
        # Run the example directly
        python mdap/example_hanoi.py "${@:2}"
    else
        # Run the Python script with all arguments
        python "$SCRIPT_DIR/../scripts/run_mdap.py" "$@"
    fi
}

# Run main function with all arguments
main "$@"
