# OVERLOOKS_FOUND.md
**Date:** 2025-10-31  
**Review:** Code review against audit report and fix plan

---

## Issues Found During Review

### 🔴 CRITICAL OVERLOOKS

#### 1. Untracked Tasks in OpenMind
**Location:** `automind/openmind.py` lines 77, 90, 104  
**Issue:** Tasks created with `asyncio.create_task()` instead of `_create_task()`  
**Impact:** Tasks not tracked for cleanup, potential resource leaks  
**Fix:** Use `_create_task()` helper method

#### 2. Conflicting Main Loop Task
**Location:** `lmagi.py` line 171  
**Issue:** `asyncio.create_task(openmind.main_loop())` runs unconditionally  
**Impact:** Conflicts with autonomous reasoning toggle, creates duplicate tasks  
**Fix:** Remove or make conditional on autonomous reasoning setting

#### 3. Dead Code - Second Entry Point
**Location:** `lmagi.py` lines 330-349  
**Issue:** Second `if __name__ == '__main__'` block will never execute  
**Impact:** Code confusion, maintenance burden  
**Fix:** Remove or document clearly

#### 4. Signal Handler References Non-Existent Method
**Location:** `lmagi.py` line 326  
**Issue:** `openmind.stop_reasoning()` doesn't exist  
**Impact:** Runtime error on shutdown  
**Fix:** Use `cleanup()` method instead

### 🟡 MEDIUM OVERLOOKS

#### 5. Unused Code in simpleeval Implementation
**Location:** `automind/logic.py` lines 143-148  
**Issue:** `operator_replacements` dictionary defined but never used  
**Impact:** Incomplete implementation, may not handle custom operators  
**Fix:** Remove unused code or implement operator replacement

#### 6. Missing Task Tracking for JavaScript Tasks
**Location:** `automind/openmind.py` line 403  
**Issue:** JavaScript task created but not tracked  
**Impact:** Minor - short-lived task, but inconsistent with tracking pattern  
**Fix:** Track if needed, or document why not tracked

---

## Fixes Required

### Priority 1 (Critical)
1. Fix untracked tasks (lines 77, 90, 104 in openmind.py)
2. Fix conflicting main_loop task (line 171 in lmagi.py)
3. Fix signal handler (line 326 in lmagi.py)
4. Remove or fix dead code entry point (lines 330-349 in lmagi.py)

### Priority 2 (Medium)
5. Clean up unused code in logic.py
6. Document or fix JavaScript task tracking

---

*End of Review*

