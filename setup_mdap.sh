#!/bin/bash

# MDAP Harness Setup Script
# Sets up environment for Massively Decomposed Agentic Processes

set -e

echo "🏗️  Setting up MDAP Harness environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv_mdap" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv_mdap
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv_mdap/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install required dependencies from requirements file
echo "📚 Installing dependencies..."
pip install -r requirements_mdap.txt

# Check if .env file exists, if not create from example
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "📝 Creating .env file from example..."
        cp .env.example .env
        echo "⚠️  IMPORTANT: Please edit .env file and add your API keys before running the examples!"
        echo ""
        echo "   Required: At least one LLM provider API key (e.g., OPENAI_API_KEY)"
        echo "   Optional: Configure alternative providers or MDAP settings"
        echo ""
    else
        echo "⚠️  No .env.example found. Creating minimal .env file..."
        cat > .env << EOF
# MDAP Environment Configuration
# Add your API keys here

OPENAI_API_KEY="your-openai-api-key-here"
LITELLM_LOG="INFO"
EOF
        echo "⚠️  Please edit .env file and add your API keys"
    fi
else
    echo "✅ .env file already exists"
fi

# Create logs directory if it doesn't exist
if [ ! -d "logs" ]; then
    echo "📁 Creating logs directory..."
    mkdir logs
fi

echo "✅ MDAP Harness setup complete!"
echo ""
echo "🚀 To get started:"
echo "   1. Activate the environment: source venv_mdap/bin/activate"
echo "   2. Set your API key in .env file"
echo "   3. Run example: python example_hanoi.py"
echo "   4. Run tests: python test_hanoi.py"
