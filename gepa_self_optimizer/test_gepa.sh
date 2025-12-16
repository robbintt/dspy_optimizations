#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Path to the requirements file, relative to the project root
REQUIREMENTS_FILE="gepa_self_optimizer/requirements.txt"

echo "=================================================================="
echo "GEPA Self Optimizer Test Runner"
echo "=================================================================="

# --- Dependency Installation ---
if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "❌ Error: requirements.txt not found at $REQUIREMENTS_FILE"
  exit 1
fi

echo "📥 Installing/upgrading dependencies from $REQUIREMENTS_FILE..."
# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r "$REQUIREMENTS_FILE"

# --- Run Tests ---
echo
echo "🧪 Running tests with pytest..."
echo "------------------------------------------------------------------"
# Run pytest on the specific test file. The 'set -e' will stop the script if tests fail.
# Discover and run all tests within the specified module/file.
python -m pytest "gepa_self_optimizer/test_gepa.py" -v
echo "------------------------------------------------------------------"

echo
echo "🏁 All steps completed successfully."
