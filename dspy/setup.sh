#!/bin/bash
set -euo pipefail

echo "--- Setting up Python virtual environment ---"
# Create virtualenvs directory if it doesn't exist
mkdir -p ~/virtualenvs

VENV_PATH="$HOME/virtualenvs/mdap_venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating Python virtual environment in ~/virtualenvs/mdap_venv..."
    python3 -m venv "$VENV_PATH"
fi
echo "Virtual environment 'mdap_venv' is ready at $VENV_PATH."


echo ""
echo "--- Installing dependencies ---"
# Activate venv for this script
source "$VENV_PATH/bin/activate"
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
echo ""
echo "--- Making run script executable ---"
chmod +x run.sh

echo ""
echo "--- Setup complete ---"
echo "Next steps:"
echo "1. Edit the .env file with your ZAI and (optional) OpenAI credentials."
echo "2. To run a script, use the run.sh helper. For example:"
echo "   ./run.sh optimize_prompt.py"
echo "   ./run.sh optimize_tool_call.py"
#!/bin/bash
set -euo pipefail

echo "--- Setting up Python virtual environment ---"
# Create virtualenvs directory if it doesn't exist
mkdir -p ~/virtualenvs

VENV_PATH="$HOME/virtualenvs/dspy_venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating Python virtual environment in ~/virtualenvs/dspy_venv..."
    python3 -m venv "$VENV_PATH"
fi
echo "Virtual environment 'dspy_venv' is ready at $VENV_PATH."


echo ""
echo "--- Installing dependencies ---"
# Activate venv for this script
source "$VENV_PATH/bin/activate"
pip install -r requirements.txt


echo ""
echo "--- Environment variables setup ---"
if [ ! -f .env ]; then
    echo "Creating .env file from root .env.example..."
    cp ../.env.example .env
    echo "IMPORTANT: Please fill in your API key and base URL in the .env file."
else
    echo ".env file already exists. Skipping creation."
fi

# Also create a symlink to root .env if it exists
if [ -f ../.env ] && [ ! -f .env ]; then
    echo "Creating symlink to root .env file..."
    ln -s ../.env .env
fi

echo ""
echo ""
echo "--- Making run script executable ---"
chmod +x run.sh

echo ""
echo "--- Setup complete ---"
echo "Next steps:"
echo "1. Edit the .env file with your ZAI and (optional) OpenAI credentials."
echo "2. To run a script, use the run.sh helper. For example:"
echo "   ./run.sh optimize_prompt.py"
echo "   ./run.sh optimize_tool_call.py"
