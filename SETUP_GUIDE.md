# lmagi Environment Setup Guide

This guide will help you set up a proper Python environment for the **lmagi** (easyAGI) project.

## Prerequisites

- **Python 3.9 or higher** (Python 3.13.7 recommended)
- **pip** package manager
- **virtualenv** (optional but recommended)

## Quick Start

### Automated Setup (Recommended)

Run the setup script to automatically create and configure your environment:

```bash
chmod +x setup.sh
./setup.sh
```

This script will:
1. Check your Python version
2. Create a virtual environment in `./venv`
3. Install all dependencies from `requirements.txt`
4. Create necessary directories
5. Create a `.env` file template for API keys

### Manual Setup

If you prefer to set up manually:

#### 1. Create Virtual Environment

```bash
python3 -m venv venv
```

#### 2. Activate Virtual Environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```cmd
venv\Scripts\activate
```

#### 3. Upgrade pip

```bash
pip install --upgrade pip
```

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 5. Create Necessary Directories

```bash
mkdir -p memory/stm memory/logs memory/truth gfx
```

#### 6. Configure API Keys

Create a `.env` file in the project root:

```bash
touch .env
```

Add your API keys to the `.env` file:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_key_here

# Groq API Key
GROQ_API_KEY=your_groq_key_here

# Together.ai API Key
TOGETHER_API_KEY=your_together_key_here

# AI71 API Key
AI71_API_KEY=your_ai71_key_here
```

## Dependencies

The project uses the following main dependencies:

### LLM API Clients
- **openai** - OpenAI API integration
- **groq** - Groq API for Llama3 models
- **together** - Together.ai API integration
- **ai71** - AI71 Falcon models

### Web Framework
- **nicegui** - UI framework with FastAPI integration
- **fastapi** - Web framework
- **uvicorn** - ASGI server

### Async & Networking
- **aiohttp** - Async HTTP client/server
- **asyncio** - Built-in async I/O (Python 3.4+)

### Utilities
- **python-dotenv** - Environment variable management
- **ujson** - Fast JSON processing
- **psutil** - System and process utilities

## Running the Application

### Main Application (Web UI)

```bash
python lmagi.py
```

This will start the web interface at `http://localhost:8080` (default NiceGUI port).

### Ollama Integration

Access the Ollama interface at:
```
http://localhost:8080/ollama
```

## Directory Structure

```
lmagi/
├── automind/           # AGI reasoning modules
│   ├── agi.py
│   ├── automind.py
│   ├── logic.py
│   ├── openmind.py
│   └── SocraticReasoning.py
├── memory/             # Memory storage
│   ├── stm/           # Short-term memory
│   ├── logs/          # Reasoning logs
│   └── truth/         # Truth tables
├── webmind/            # Web and API handlers
│   ├── api.py
│   ├── chatter.py
│   ├── html_head.py
│   └── ollama_handler.py
├── gfx/                # Static assets
├── lmagi.py            # Main application
├── requirements.txt    # Dependencies
├── setup.sh           # Setup script
├── setup.py           # Package configuration
└── .env               # API keys (create this)
```

## Troubleshooting

### Virtual Environment Not Activating

**Issue:** `source venv/bin/activate` doesn't work

**Solution:** Make sure you're in the project root directory and the virtual environment was created successfully.

### Missing Dependencies

**Issue:** Import errors when running the application

**Solution:** Ensure you've activated the virtual environment and installed all dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Port Already in Use

**Issue:** NiceGUI can't start because port is in use

**Solution:** Either stop the process using the port or specify a different port in the code.

### API Key Errors

**Issue:** Authentication errors with LLM APIs

**Solution:**
1. Verify your API keys are correct in the `.env` file
2. Ensure there are no extra spaces or quotes around the keys
3. Check that the `.env` file is in the project root directory

### asyncio Module Error

**Issue:** `ModuleNotFoundError: No module named 'asyncio'`

**Solution:** The `asyncio` module is built into Python 3.4+. If you see this error:
1. Check your Python version: `python3 --version`
2. Remove the `asyncio==3.4.3` line from requirements.txt (it's not needed for Python 3.9+)

## Development

### Installing Development Dependencies

Uncomment the development dependencies in `requirements.txt`:

```txt
pytest>=7.4.0  # Testing framework
black>=23.0.0  # Code formatter
flake8>=6.0.0  # Linter
mypy>=1.5.0    # Type checker
```

Then install:
```bash
pip install -r requirements.txt
```

### Code Formatting

```bash
black .
```

### Linting

```bash
flake8 .
```

### Type Checking

```bash
mypy .
```

## Package Installation

To install lmagi as a package:

```bash
pip install -e .
```

This allows you to import lmagi modules from anywhere in your system while in development mode.

## Support

- **Documentation:** https://rage.pythai.net
- **License:** MIT License (c) 2024 Gregory L. Magnusson

## Notes

- The application requires at least one API key to function
- Memory files will be created automatically on first run
- All conversation data is stored locally in the `memory` directory
- The autonomous reasoning feature can be enabled/disabled via the UI toggle
