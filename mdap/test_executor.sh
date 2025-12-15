#!/bin/bash

# Test script for MicroAgentExecutor
# Tests basic functionality of the generic microagent calling class

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${HOME}/virtualenvs/mdap_harness_venv"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "\033[0;31m[ERROR]\033[0m Virtual environment not found!"
    echo "Please run './setup_mdap.sh' first to set up the environment."
    exit 1
fi

echo -e "\033[0;34m[Test]\033[0m Activating virtual environment..."
source "${VENV_PATH}/bin/activate"

# Change to script directory to ensure tests run from correct location
cd "$SCRIPT_DIR"

echo ""
echo -e "\033[0;34m[Test]\033[0m Running basic executor tests..."
python test_executor_basic.py

echo ""
echo -e "\033[0;34m[Test]\033[0m Running unit tests..."
python -m pytest test_micro_agent_executor.py -v

echo ""
echo -e "\033[0;32m[Test]\033[0m All executor tests passed!"
