#!/bin/bash

# calibrate_hanoi.sh
# Script to run the calibration for the Hanoi solver.

# Check if a number of disks was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <num_disks>"
    echo "Example: $0 2"
    exit 1
fi

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$HOME/virtualenvs/mdap_harness_venv"

# Activate virtual environment
echo "[MDAP] Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Run the Python calibration script
echo "[MDAP] Running calibration for $1 disks..."
cd "$SCRIPT_DIR"
python calibrate_hanoi.py "$1"

echo "[SUCCESS] Done!"
