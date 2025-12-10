#!/bin/bash

# calibrate_hanoi.sh
# Script to run the calibration for the Hanoi solver.

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$HOME/virtualenvs/mdap_harness_venv"

# Activate virtual environment
echo "[MDAP] Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Run the Python calibration script
# Default to 20 disks if no argument provided
DISKS=${1:-20}
echo "[MDAP] Running calibration for $DISKS disks..."
cd "$SCRIPT_DIR"
python calibrate_hanoi.py --sample_steps "$DISKS" "$@"

echo "[SUCCESS] Done!"
