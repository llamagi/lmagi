#!/bin/bash
# setup.sh - Python Environment Setup Script for lmagi
# (c) Gregory L. Magnusson MIT license 2024

set -e  # Exit on error

echo "=========================================="
echo "lmagi Python Environment Setup"
echo "=========================================="

# Check Python version
REQUIRED_PYTHON="3.9"
PYTHON_CMD="python3"

echo "Checking Python version..."
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at ./$VENV_DIR"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment..."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "Virtual environment created at ./$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p memory/stm
mkdir -p memory/logs
mkdir -p memory/truth
mkdir -p gfx

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
# API Keys Configuration
# Add your API keys below

# OpenAI API Key
# OPENAI_API_KEY=your_openai_key_here

# Groq API Key
# GROQ_API_KEY=your_groq_key_here

# Together.ai API Key
# TOGETHER_API_KEY=your_together_key_here

# AI71 API Key
# AI71_API_KEY=your_ai71_key_here
EOF
    echo ".env file created. Please add your API keys to it."
else
    echo ".env file already exists."
fi

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
pip list | grep -E "openai|groq|together|ai71|nicegui|aiohttp|asyncio|ujson|psutil|python-dotenv"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the application:"
echo "  python lmagi.py"
echo ""
echo "Don't forget to add your API keys to the .env file!"
echo "=========================================="
