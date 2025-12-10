#!/bin/bash

# calibrate_hanoi.sh
# Script to run the calibration for the Hanoi solver.

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$HOME/virtualenvs/mdap_harness_venv"

# Find parent directory with .env file
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PARENT_DIR/.env"

# Check if .env file exists (optional)
if [ -f "$ENV_FILE" ]; then
    echo "[MDAP] Found .env file at $ENV_FILE"
    # Export .env file path for Python scripts
    export ENV_FILE_PATH="$ENV_FILE"
else
    echo "[MDAP] No .env file found at $ENV_FILE, using system defaults"
fi

# Activate virtual environment
echo "[MDAP] Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Run the Python calibration script
echo "[MDAP] Running calibration..."
echo "[MDAP] Note: Cache regeneration is now the default. Use --use_cache to skip."
cd "$SCRIPT_DIR"
# Execute Python script with all arguments intact
exec python calibrate_hanoi.py "$@"

echo "[SUCCESS] Done!"
echo "[INFO] To analyze the results, run: ./mdap/extract_calibration_data.py"
echo "[INFO] The full report will be saved to: docs/analysis/calibration_data_analysis.md"
