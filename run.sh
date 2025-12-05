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

# Create a logs directory if it doesn't exist
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Generate a log file name based on the script and current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/${SCRIPT_NAME%.py}_${TIMESTAMP}.log"

echo ""
echo "--- Running python $SCRIPT_NAME ---"
echo "--- Output will be logged to $LOG_FILE ---"

# Use python's -u flag for unbuffered output.
# Redirect stderr (2) to stdout (1) so both are captured.
# `tee` writes the output to the specified log file AND to the console.
python -u "$SCRIPT_NAME" 2>&1 | tee "$LOG_FILE"

echo ""
echo "--- Script finished. Log saved to: $LOG_FILE ---"
