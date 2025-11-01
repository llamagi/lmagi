# LMAGI Fix Plan
**Based on Audit Report**  
**Date:** 2025-10-31

---

## Fix Plan Overview

This plan addresses all critical and high-priority issues identified in the audit report, organized by priority and dependency.

---

## Phase 1: Critical Fixes (🔴 IMMEDIATE)

### Fix 1.1: OpenAI API Syntax Error
**File:** `webmind/chatter.py`  
**Line:** 31  
**Priority:** 🔴 CRITICAL  
**Estimated Time:** 15 minutes

**Current Code:**
```python
response = openai.chatcompletion.create(
    model=self.current_model,
    messages=[...]
)
```

**Fix:**
```python
from openai import OpenAI

class GPT4o:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.current_model = "gpt-4o"
    
    def generate_response(self, knowledge):
        prompt = f"{knowledge}"
        try:
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                ]
            )
            decision = response.choices[0].message.content
            return decision.lower()
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return f"error: unable to generate a response due to an issue with the openai api."
```

**Dependencies:** None  
**Testing:** Test with OpenAI API key

---

### Fix 1.2: Missing Attribute Initialization
**File:** `automind/openmind.py`  
**Line:** 36, 41  
**Priority:** 🔴 CRITICAL  
**Estimated Time:** 5 minutes

**Current Code:**
```python
def __init__(self):
    # Missing: self.autonomous_reasoning = False
    # Missing: self.reasoning_task = None
```

**Fix:**
```python
def __init__(self):
    self.api_manager = APIManager()
    self.agi_instance = None
    self.initialize_memory()
    self.message_container = ui.column()
    self.ollama_handler = OllamaHandler()
    self.internal_queue = asyncio.Queue()
    self.prompt = ""
    self.keys_container = ui.column()
    self.log = None
    self.initialization_warning_shown = False
    self.autonomous_reasoning = False  # ADD THIS
    self.reasoning_task = None  # ADD THIS
```

**Dependencies:** None  
**Testing:** Toggle autonomous reasoning in UI

---

### Fix 1.3: Duplicate Function Definition
**File:** `lmagi.py`  
**Lines:** 238, 280  
**Priority:** 🔴 CRITICAL  
**Estimated Time:** 10 minutes

**Issue:** `select_ollama_model()` defined twice

**Fix:** Remove duplicate definition at line 280, keep the one at line 238

**Dependencies:** None  
**Testing:** Test Ollama model selection

---

### Fix 1.4: Import Error
**File:** `automind/agi.py`  
**Line:** 8  
**Priority:** 🔴 CRITICAL  
**Estimated Time:** 2 minutes

**Current Code:**
```python
from webmind.chatter import GPT4o, Groq
```

**Fix:**
```python
from webmind.chatter import GPT4o, GroqModel
```

**Dependencies:** None  
**Testing:** Verify imports work

---

### Fix 1.5: Logic String Comparison Bug
**File:** `automind/openmind.py`  
**Line:** 304  
**Priority:** 🔴 CRITICAL  
**Estimated Time:** 5 minutes

**Issue:** String mismatch between return value and comparison

**Current Code:**
```python
# Line 197 (SocraticReasoning.py):
return "No premises available for logic as conclusion."  # With period

# Line 304 (openmind.py):
if conclusion == "No premises available for logic as conclusion.":  # With period
```

**Fix:** Standardize to one format (preferably without period):
```python
# In SocraticReasoning.py line 197:
return "No premises available for logic as conclusion"  # Remove period

# In openmind.py line 304:
if conclusion == "No premises available for logic as conclusion":  # Remove period
```

**Dependencies:** None  
**Testing:** Test with empty premises

---

## Phase 2: High Priority Fixes (🟡 HIGH)

### Fix 2.1: eval() Security Risk
**File:** `automind/logic.py`  
**Line:** 136  
**Priority:** 🟡 HIGH (Security)  
**Estimated Time:** 2-4 hours

**Current Code:**
```python
result = eval(expr, {"__builtins__": None}, {**allowed_operators, **values})
```

**Fix Options:**

**Option A:** Use AST literal eval (safer but limited)
```python
import ast

def safe_eval_expression(expr, values):
    # Only allow simple variable references and operators
    # This is complex - may need custom parser
    pass
```

**Option B:** Use a proper expression evaluator library
```python
# Install: pip install simpleeval
from simpleeval import simple_eval

def evaluate_expression(self, expr, values):
    try:
        # simple_eval is safer than eval
        result = simple_eval(expr, names=values, functions=allowed_operators)
        return result
    except Exception as e:
        self.log(f"Error evaluating expression '{expr}': {e}", level='error')
        return False
```

**Option C:** Create a custom safe evaluator
- Parse expression into AST
- Validate only allowed operators
- Execute validated AST

**Recommendation:** Option B (simpleeval) - fastest and safest

**Dependencies:** Add `simpleeval` to requirements.txt  
**Testing:** Test with various expressions, including malicious ones

---

### Fix 2.2: Async/Sync Pattern Issues
**File:** `webmind/chatter.py`  
**Line:** 110  
**Priority:** 🟡 HIGH  
**Estimated Time:** 1 hour

**Current Code:**
```python
def generate_response(self, knowledge):
    return asyncio.run(self.generate_response_async(knowledge, self.current_model))
```

**Fix:** Make all chatter methods async-compatible:
```python
async def generate_response_async(self, knowledge, model=None):
    if model is None:
        model = self.current_model
    # ... existing async code ...

# For sync compatibility when needed:
def generate_response(self, knowledge):
    """Synchronous wrapper - use only when not in async context"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to use run_in_executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.generate_response_async(knowledge))
                return future.result()
        else:
            return loop.run_until_complete(self.generate_response_async(knowledge))
    except RuntimeError:
        return asyncio.run(self.generate_response_async(knowledge))
```

**Dependencies:** Review all call sites  
**Testing:** Test in both sync and async contexts

---

### Fix 2.3: Add Timeout/Retry Logic
**Files:** `webmind/chatter.py`, `webmind/ollama_handler.py`  
**Priority:** 🟡 HIGH  
**Estimated Time:** 2 hours

**Fix:** Add timeout and retry decorator:
```python
import asyncio
from functools import wraps
from typing import TypeVar, Callable

T = TypeVar('T')

def retry_with_timeout(max_retries=3, timeout=30.0, backoff=1.0):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                except (asyncio.TimeoutError, Exception) as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(backoff * (attempt + 1))
            return None
        return wrapper
    return decorator

# Apply to API calls:
@retry_with_timeout(max_retries=3, timeout=30.0)
async def generate_response_async(self, knowledge, model=None):
    # ... existing code ...
```

**Dependencies:** None  
**Testing:** Test with network failures, slow connections

---

### Fix 2.4: Task Cleanup and Management
**File:** `automind/openmind.py`  
**Priority:** 🟡 HIGH  
**Estimated Time:** 1 hour

**Current Issues:**
- Tasks not tracked
- No cleanup on shutdown
- Tasks can be overwritten

**Fix:**
```python
class OpenMind:
    def __init__(self):
        # ... existing code ...
        self.tasks = set()  # Track all tasks
        
    async def create_task(self, coro):
        """Create and track a task"""
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task
    
    async def cleanup(self):
        """Cancel all tracked tasks"""
        for task in self.tasks:
            if not task.done():
                task.cancel()
        # Wait for cancellation
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
```

**Dependencies:** None  
**Testing:** Test task cancellation, shutdown

---

### Fix 2.5: Standardize Error Handling
**Files:** All  
**Priority:** 🟡 HIGH  
**Estimated Time:** 3 hours

**Fix:** Create custom exception classes:
```python
# webmind/exceptions.py
class LmagiException(Exception):
    """Base exception for lmagi"""
    pass

class APIError(LmagiException):
    """API-related errors"""
    pass

class ReasoningError(LmagiException):
    """Reasoning engine errors"""
    pass

class MemoryError(LmagiException):
    """Memory storage errors"""
    pass
```

Update all error handling to use these exceptions.

**Dependencies:** None  
**Testing:** Test error scenarios

---

## Phase 3: Medium Priority Improvements (🟢 MEDIUM)

### Fix 3.1: Extract Magic Numbers to Configuration
**Files:** Multiple  
**Priority:** 🟢 MEDIUM  
**Estimated Time:** 1 hour

**Create:** `config.py`
```python
# Configuration constants
REASONING_LOOP_INTERVAL = 10  # seconds
RETRY_WAIT_TIME = 30  # seconds
MAX_PREMISES = 5
SERVER_PORT = 8080
OLLAMA_API_URL = "http://localhost:11434/api"
MAX_TOKENS_DEFAULT = 100
```

**Dependencies:** None  
**Testing:** Verify all references updated

---

### Fix 3.2: Add Type Hints
**Files:** All  
**Priority:** 🟢 MEDIUM  
**Estimated Time:** 4-6 hours

**Start with:**
- Public API methods
- Function parameters
- Return types

**Example:**
```python
from typing import Optional, List, Dict, Any

def generate_response(self, knowledge: str) -> str:
    # ...
```

**Dependencies:** None  
**Testing:** Type checking with mypy

---

### Fix 3.3: File Path Validation
**File:** `memory/memory.py`  
**Priority:** 🟢 MEDIUM  
**Estimated Time:** 1 hour

**Fix:**
```python
import os
from pathlib import Path

def validate_path(file_path: str) -> bool:
    """Validate file path is safe"""
    try:
        resolved = Path(file_path).resolve()
        # Ensure path is within memory directory
        memory_dir = Path('./memory').resolve()
        return str(resolved).startswith(str(memory_dir))
    except Exception:
        return False
```

**Dependencies:** None  
**Testing:** Test with malicious paths

---

### Fix 3.4: JSON Error Handling
**Files:** Multiple  
**Priority:** 🟢 MEDIUM  
**Estimated Time:** 2 hours

**Fix:** Add try/except around all JSON operations:
```python
def safe_json_load(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading JSON from {file_path}: {e}")
        return []  # or appropriate default
```

**Dependencies:** None  
**Testing:** Test with corrupted JSON files

---

## Phase 4: Testing and Quality (⚪ LOW)

### Fix 4.1: Add Unit Tests
**Priority:** ⚪ LOW  
**Estimated Time:** 8-12 hours

**Create test structure:**
```
tests/
├── test_chatter.py
├── test_memory.py
├── test_reasoning.py
├── test_openmind.py
└── conftest.py
```

**Dependencies:** Install pytest  
**Testing:** Run test suite

---

### Fix 4.2: Add Integration Tests
**Priority:** ⚪ LOW  
**Estimated Time:** 4-6 hours

**Test:**
- End-to-end user flows
- API integration
- Memory persistence

**Dependencies:** Unit tests complete  
**Testing:** Full integration test suite

---

## Implementation Order

### Week 1: Critical Fixes
1. ✅ Fix 1.1: OpenAI API syntax
2. ✅ Fix 1.2: Missing attributes
3. ✅ Fix 1.3: Duplicate function
4. ✅ Fix 1.4: Import error
5. ✅ Fix 1.5: String comparison

### Week 2: High Priority
1. Fix 2.1: eval() security
2. Fix 2.2: Async/sync patterns
3. Fix 2.3: Timeout/retry logic
4. Fix 2.4: Task cleanup
5. Fix 2.5: Error handling

### Week 3: Medium Priority
1. Fix 3.1: Configuration
2. Fix 3.2: Type hints (partial)
3. Fix 3.3: Path validation
4. Fix 3.4: JSON handling

### Week 4+: Testing and Polish
1. Unit tests
2. Integration tests
3. Documentation
4. Performance optimization

---

## Risk Mitigation

### Before Each Fix:
1. Create git branch
2. Write test case (if applicable)
3. Implement fix
4. Test locally
5. Run existing tests (if any)
6. Code review
7. Merge to main

### After Phase 1:
- Re-run audit
- Verify all critical issues resolved
- Test with real API keys
- Monitor logs for errors

---

## Success Criteria

### Phase 1 Complete When:
- ✅ All 🔴 CRITICAL bugs fixed
- ✅ OpenAI API works
- ✅ No runtime errors on startup
- ✅ Basic functionality works

### Phase 2 Complete When:
- ✅ All 🟡 HIGH priority issues resolved
- ✅ Security risks mitigated
- ✅ Error handling standardized
- ✅ Async patterns consistent

### Phase 3 Complete When:
- ✅ Code quality improved
- ✅ Configuration centralized
- ✅ Type hints added (partial)
- ✅ Path validation working

---

## Notes

- Each fix should be atomic and testable
- Don't fix multiple issues in one commit
- Write tests before fixing when possible
- Document changes in commit messages
- Update this plan as fixes are completed

---

*End of Fix Plan*

