# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**lmagi** (language model Augmented Generative Intelligence) is an experimental AGI framework that combines multi-model LLM integration with Socratic reasoning and memory systems. The project implements autonomous reasoning loops with premise-based logical inference, storing reasoning artifacts in a structured memory system.

## Quick Start Commands

### Environment Setup
```bash
# Initial setup (creates venv, installs deps, creates .env)
./setup.sh

# Activate environment
source venv/bin/activate

# Check environment status
./manage.sh check

# Run tests on installation
./manage.sh test
```

### Running the Application
```bash
# Main web UI (runs on http://localhost:8080)
python lmagi.py
# or
./manage.sh run

# Ollama integration page available at /ollama route
```

### Development
```bash
# Update dependencies
./manage.sh update

# Install dev dependencies
pip install -r requirements-dev.txt
```

## Core Architecture

### Multi-Layer Reasoning System

The application has three distinct reasoning layers that work together:

1. **User Interaction Layer** (`lmagi.py`, `openmind.py`)
   - NiceGUI-based web interface with FastAPI backend
   - Handles user input/output via chat interface
   - Manages API key configuration and model selection
   - Routes to main chat (`/`) and Ollama interface (`/ollama`)

2. **AGI Reasoning Layer** (`automind/`)
   - `openmind.py`: Orchestrates two concurrent loops:
     - `main_loop()`: Processes user input from queue
     - `reasoning_loop()`: Autonomous internal reasoning (10s intervals)
   - `automind.py`: `FundamentalAGI` class wraps AGI with chatter models
   - `agi.py`: Core `AGI` class initializes `SocraticReasoning`

3. **Socratic Reasoning Engine** (`automind/SocraticReasoning.py`)
   - Implements premise-based logical reasoning
   - `add_premise()`: Validates and stores premises
   - `draw_conclusion()`: Generates up to 5 additional premises, validates through logic tables
   - `challenge_premise()`: Removes premises and equivalent premises
   - Uses `LogicTables` for tautology validation

### Memory System Architecture

Memory is hierarchical and persistent across sessions:

**Directory Structure:**
- `memory/stm/` - Short-term memory: timestamped user conversations as `{timestamp}memory.json`
- `memory/logs/` - Reasoning logs:
  - `premises.json` - Valid premises with conclusions
  - `notpremise.json` - Invalid premises/conclusions
  - `thoughts.json` - Internal reasoning conclusions
  - `conclusions.txt` - Appended log of all conclusions
  - `socraticlogs.txt` - Detailed Socratic reasoning logs
  - `truth.json` - Truth table data
- `memory/truth/` - Logical truths:
  - `belief_{timestamp}.json` - Logic table states
  - `{timestamp}_truth.json` - Validated truths
  - `logs.txt` - Truth table logs
- `mindx/` - Internal reasoning artifacts:
  - `{timestamp}internalmemory.json` - Valid internal conclusions
  - `nopremise{timestamp}internalmemory.json` - Empty premise cycles

**Memory Functions** (`memory/memory.py`):
- `create_memory_folders()`: Ensures all directories exist
- `store_in_stm(DialogEntry)`: Saves user conversations
- `save_conversation_memory()`: Saves instruction-response pairs
- `save_internal_reasoning()`: Saves autonomous reasoning outputs
- `save_valid_truth()`: Persists validated logical truths

### Multi-Model LLM Integration

**Supported Providers** (`webmind/chatter.py`):
- OpenAI: `GPT4o` class (default: gpt-4o)
- Groq: `GroqModel` class (default: mixtral-8x7b-32768)
- Together.ai: `TogetherModel` class (default: mistralai/Mixtral-8x7B-Instruct-v0.1)
- AI71: `AI71Model` class (default: tiiuae/falcon-180B-chat)
- Ollama: `OllamaHandler` class (local models via http://localhost:11434)

**Model Selection Flow:**
1. User selects provider via FAB menu in UI
2. `OpenMind.select_model()` initializes appropriate chatter
3. Chatter instance passed to `FundamentalAGI(chatter)`
4. All reasoning uses selected model's `generate_response()` method

**API Management** (`webmind/api.py`):
- `APIManager` loads keys from `.env` file
- Format: `{SERVICE}_API_KEY` (e.g., `OPENAI_API_KEY`)
- UI allows runtime add/delete/list of API keys
- Keys saved to `.env` via `python-dotenv.set_key()`

### Logic and Truth Tables

**Logic System** (`automind/logic.py`):
- `LogicTables` class implements propositional logic
- Supports operators: and, or, not, xor, nand, nor, implication
- `generate_truth_table()`: Creates all combinations for variables
- `tautology(expression)`: Validates if expression is always true
- `modus_ponens(fact1, fact2)`: Applies logical inference rule
- All logic operations logged to `memory/truth/logs.txt`

**Truth Storage:**
- Beliefs: Variable/expression additions logged
- Truths: Validated tautologies with timestamps
- Facts: Modus ponens conclusions
- Contingent: Non-tautology expressions

## Critical Implementation Details

### Asynchronous Execution Pattern

The system uses dual concurrent tasks that must not block each other:

```python
# In openmind.py main_loop()
reasoning_task = asyncio.create_task(self.reasoning_loop())  # Background
while True:
    prompt = await self.internal_queue.get()  # User input queue
    conclusion = await self.get_conclusion_from_agi(prompt)
```

**Important:**
- `reasoning_loop()` runs independently every 10 seconds
- User input processed via `internal_queue` (asyncio.Queue)
- Both use `run_in_executor()` to avoid blocking on synchronous LLM calls

### UI Context Management

NiceGUI requires careful context handling:

```python
# Always check client connection before UI operations
if self.message_container.client.connected:
    with self.message_container:
        ui.notify('message')
```

**Common Pattern:**
1. Check `.client.connected` before any UI update
2. Use `with container:` context manager for element addition
3. `.clear()` before replacing container contents

### Memory Persistence Flow

Every interaction creates multiple memory artifacts:

1. **User sends message:**
   - Saved to `stm/{timestamp}memory.json` via `store_in_stm()`
   - Added to `internal_queue` for processing

2. **AGI processes (via `draw_conclusion()`):**
   - Premises saved to `logs/premises.json`
   - Invalid premises → `logs/notpremise.json`
   - Conclusion appended to `logs/conclusions.txt`
   - Valid truths → `truth/{timestamp}_truth.json`

3. **Autonomous reasoning (every 10s):**
   - Internal conclusions → `logs/thoughts.json`
   - No premises → `logs/notpremise.json`
   - Also saved to `mindx/{timestamp}internalmemory.json`

### Ollama Integration

**Key Differences from API models:**
- Ollama runs locally, checked via `httpx.get('http://localhost:11434')`
- Uses streaming responses: `stream=True` in payload
- Response chunks parsed line-by-line with ujson
- Model list obtained via subprocess: `ollama list`
- Separate UI route at `/ollama` with dedicated model selector

**Note:** Ollama is NOT yet integrated with SocraticReasoning/AGI components (per README).

## Configuration Requirements

### Required API Keys (.env)

At least ONE of these is required:
```env
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
TOGETHER_API_KEY=...
AI71_API_KEY=...
```

### Python Version

- **Minimum:** Python 3.9
- **Recommended:** Python 3.13.7 (specified in `.python-version`)

## Important Caveats

### Known Limitations

1. **Ollama Not Fully Integrated:** While Ollama UI exists at `/ollama`, it's not connected to the AGI reasoning components
2. **Autonomous Reasoning:** The `reasoning_loop()` uses the last user prompt repeatedly - may generate redundant conclusions
3. **asyncio Installation:** Do NOT install `asyncio` package - it's built into Python 3.4+. Remove from requirements if present.
4. **Model Token Limits:** SocraticReasoning defaults to `max_tokens=100` for premise generation

### File Paths

All file operations use relative paths from project root:
- Memory: `./memory/stm/`, `./memory/logs/`, etc.
- Static assets: `./gfx/` (mounted to `/gfx` in web server)
- Logs must use pathlib.Path for cross-platform compatibility

## Testing Approach

No formal test suite exists yet. Manual testing via:
```bash
# Test installation
./manage.sh test

# Test with sample interaction
python lmagi.py
# Navigate to http://localhost:8080
# Add API key via APIk tab
# Select model from FAB menu
# Send test prompt
# Check memory/stm/ and memory/logs/ for artifacts
```

## Dependencies Note

**Critical packages:**
- `nicegui` (UI framework, includes FastAPI/Uvicorn)
- `aiohttp` (async HTTP for Ollama)
- `ujson` (faster JSON for memory operations)
- `python-dotenv` (API key management)
- Provider SDKs: `openai`, `groq`, `together`, `ai71`

**Do NOT use:** `asyncio` package (use built-in Python module instead)
