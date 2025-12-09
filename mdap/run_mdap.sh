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

# Function to check if .env file exists
check_env() {
    if [ ! -f ".env" ]; then
        echo -e "\033[1;33m[WARNING]\033[0m .env file not found!"
        echo "Please create a .env file with your API keys."
        echo "You can copy it from .env.example:"
        echo "  cp .env.example .env"
        echo "Then edit .env with your API keys."
        exit 1
    fi
}

# Main script logic
main() {
    # Check prerequisites
    check_venv
    activate_venv
    check_env
    
    # Run the Python script with all arguments
    python ../scripts/run_mdap.py "$@"
}

# Run main function with all arguments
main "$@"
