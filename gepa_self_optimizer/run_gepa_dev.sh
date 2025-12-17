#!/bin/bash

# A convenience wrapper to activate venv and run the GEPA optimizer
# using the development configuration (models.dev.yaml).

set -e # Exit immediately if a command exits with a non-zero status.

# --- Resolve Project Root ---
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Get the project root directory (one level up from the script's directory)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# --- Configuration ---
VENV_DIR="$PROJECT_ROOT/venv"
CONFIG_DIR="$SCRIPT_DIR/config"
MODELS_FILE="$CONFIG_DIR/models.yaml"
DEV_MODELS_FILE="$CONFIG_DIR/models.dev.yaml"
BACKUP_FILE="$CONFIG_DIR/models.yaml.backup"

# --- Script Logic ---
echo "🚀 Starting GEPA development run..."

# 1. Check for venv
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Error: Virtual environment '$VENV_DIR' not found."
    echo "Please create it first: python -m venv venv"
    exit 1
fi

# 2. Activate venv
echo "🔧 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 3. Check if API key is set
if [ -z "$CEREBRAS_API_KEY" ] && [ -z "$ZHIPUAI_API_KEY" ]; then
    echo "⚠️ Warning: Neither CEREBRAS_API_KEY nor ZHIPUAI_API_KEY is set."
    echo "The optimizer may fail. Press Ctrl+C to cancel, or Enter to continue..."
    read -r
fi

# 4. Switch to development config
echo "🔄 Switching to development configuration..."

if [ -f "$MODELS_FILE" ]; then
    mv "$MODELS_FILE" "$BACKUP_FILE"
    echo "🔹 Backed up existing '$MODELS_FILE' to '$BACKUP_FILE'."
fi

cp "$DEV_MODELS_FILE" "$MODELS_FILE"
echo "🔹 Using development models from '$DEV_MODELS_FILE'."

# 5. Run the scripts
echo "📦 Step 1: Generating synthetic data..."
python "$PROJECT_ROOT/gepa_self_optimizer/generate_data.py"

echo "🧬 Step 2: Running GEPA optimization..."
python "$PROJECT_ROOT/gepa_self_optimizer/optimize_gepa.py"

# 6. Restore original config
echo "🔄 Restoring original configuration..."
cd "$CONFIG_DIR"
if [ -f "$BACKUP_FILE" ]; then
    mv "$BACKUP_FILE" "$MODELS_FILE"
    echo "🔹 Restored original '$MODELS_FILE' from backup."
else
    # If there was no original, remove the dev config we copied
    rm "$MODELS_FILE"
    echo "🔹 Removed temporary development config."
fi

# 7. Deactivate venv
echo "🔧 Deactivating virtual environment..."
deactivate

echo "✅ GEPA development run complete!"
echo "Results are in 'gepa_self_optimizer/golden_set.json' and 'gepa_self_optimizer/glm_gepa_complete.json'"
