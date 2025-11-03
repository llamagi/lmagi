# lmagi Quick Start

## 🚀 Fast Setup (30 seconds)

```bash
# 1. Run the setup script
./setup.sh

# 2. Activate virtual environment
source venv/bin/activate

# 3. Add your API keys to .env file
nano .env  # or use your preferred editor

# 4. Run the application
python lmagi.py
```

## 📋 What You Need

At least ONE of these API keys:
- OpenAI API key → https://platform.openai.com/api-keys
- Groq API key → https://console.groq.com/keys
- Together.ai API key → https://api.together.xyz/settings/api-keys
- AI71 API key → https://marketplace.ai71.ai/

## 🔧 Essential Commands

```bash
# Activate environment
source venv/bin/activate

# Deactivate environment
deactivate

# Run main app
python lmagi.py

# Install new package
pip install package_name

# Update requirements
pip freeze > requirements.txt

# Check installed packages
pip list
```

## 🌐 URLs

- Main UI: http://localhost:8080
- Ollama Interface: http://localhost:8080/ollama

## 📁 Important Files

- `.env` - Your API keys (KEEP SECRET!)
- `requirements.txt` - Python dependencies
- `memory/` - Conversation storage
- `lmagi.py` - Main application

## ⚡ Features

- **Multi-Model Support**: OpenAI, Groq, Together.ai, AI71, Ollama
- **Autonomous Reasoning**: Toggle on/off in UI
- **Memory System**: Automatic conversation logging
- **Truth Tables**: Logical reasoning tracking
- **Dark Mode**: Toggle in UI

## 🆘 Quick Fixes

**Can't activate venv?**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Missing dependencies?**
```bash
pip install -r requirements.txt
```

**API errors?**
```bash
# Check your .env file has correct keys
cat .env
```

## 📚 Full Documentation

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed setup instructions.
