#!/bin/bash
set -euo pipefail

echo "--- Setting up Python virtual environment ---"
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment 'venv'..."
    python3 -m venv venv
fi
echo "Virtual environment 'venv' is ready."


echo ""
echo "--- Installing dependencies ---"
# Activate venv for this script
source venv/bin/activate
pip install -r requirements.txt


echo ""
echo "--- Environment variables setup ---"
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "IMPORTANT: Please fill in your API key and base URL in the .env file."
else
    echo ".env file already exists. Skipping creation."
fi

echo ""
echo "--- Setup complete ---"
echo "Next steps:"
echo "1. Edit the .env file with your credentials."
echo "2. To run the script, use the following commands:"
echo "   source venv/bin/activate"
echo "   source .env"
echo "   python optimize_prompt.py"
