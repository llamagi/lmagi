#!/usr/bin/env python3
"""
System Verification Script for lmagi
Verifies autonomous mode, logging systems, and memory writing
"""

import os
import sys
import json
import pathlib
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_memory_folders():
    """Verify all memory folders exist"""
    print("\n" + "="*60)
    print("CHECKING MEMORY FOLDERS")
    print("="*60)
    
    folders = [
        "./memory/",
        "./memory/stm/",
        "./memory/ltm/",
        "./memory/episodic/",
        "./memory/truth/",
        "./memory/logs/",
        "./mindx/",
        "./mindx/agency/",
        "./mindx/errors/"
    ]
    
    all_exist = True
    for folder in folders:
        exists = os.path.exists(folder)
        status = "✓" if exists else "✗"
        print(f"{status} {folder}")
        if not exists:
            all_exist = False
            print(f"  WARNING: Folder does not exist!")
    
    return all_exist

def check_log_files():
    """Verify log files exist and are writable"""
    print("\n" + "="*60)
    print("CHECKING LOG FILES")
    print("="*60)
    
    log_files = {
        "Premises": "./memory/logs/premises.json",
        "Not Premise": "./memory/logs/notpremise.json",
        "Thoughts": "./memory/logs/thoughts.json",
        "Conclusions": "./memory/logs/conclusions.txt",
        "Truth Tables": "./memory/logs/truth.json",
        "Socratic Logs": "./memory/logs/socraticlogs.txt",
        "Truth Logs": "./memory/truth/logs.txt"
    }
    
    all_ok = True
    for name, path in log_files.items():
        exists = os.path.exists(path)
        writable = os.access(os.path.dirname(path), os.W_OK) if os.path.exists(os.path.dirname(path)) else False
        
        status = "✓" if (exists and writable) else "✗"
        print(f"{status} {name}: {path}")
        
        if exists:
            try:
                size = os.path.getsize(path)
                print(f"    Size: {size} bytes")
                
                # Try to read if JSON
                if path.endswith('.json'):
                    with open(path, 'r') as f:
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                print(f"    Entries: {len(data)}")
                            elif isinstance(data, dict):
                                print(f"    Keys: {list(data.keys())[:5]}")
                        except json.JSONDecodeError:
                            # Check if it's line-delimited JSON
                            with open(path, 'r') as f2:
                                lines = f2.readlines()
                                print(f"    Lines: {len(lines)}")
            except Exception as e:
                print(f"    Error reading: {e}")
        
        if not writable:
            print(f"    WARNING: Directory not writable!")
            all_ok = False
    
    return all_ok

def check_truth_table_logging():
    """Verify truth table logging functionality"""
    print("\n" + "="*60)
    print("CHECKING TRUTH TABLE LOGGING")
    print("="*60)
    
    try:
        from automind.logic import LogicTables
        
        lt = LogicTables()
        print("✓ LogicTables initialized")
        
        # Test adding variables
        lt.add_variable('A')
        lt.add_variable('B')
        print("✓ Variables added")
        
        # Test adding expressions
        lt.add_expression('A and B')
        print("✓ Expressions added")
        
        # Check if belief files are created
        truth_dir = "./memory/truth"
        belief_files = list(pathlib.Path(truth_dir).glob("*_belief.json"))
        print(f"✓ Belief files found: {len(belief_files)}")
        
        # Check truth logs
        truth_log_path = "./memory/truth/logs.txt"
        if os.path.exists(truth_log_path):
            with open(truth_log_path, 'r') as f:
                lines = f.readlines()
                print(f"✓ Truth logs: {len(lines)} lines")
        
        return True
    except Exception as e:
        print(f"✗ Error checking truth tables: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_socratic_reasoning():
    """Verify SocraticReasoning logging"""
    print("\n" + "="*60)
    print("CHECKING SOCRATIC REASONING LOGGING")
    print("="*60)
    
    try:
        from automind.SocraticReasoning import SocraticReasoning
        
        # Check if files exist
        files = {
            "Premises": "./memory/logs/premises.json",
            "Not Premise": "./memory/logs/notpremise.json",
            "Truth Tables": "./memory/logs/truth.json",
            "Conclusions": "./memory/logs/conclusions.txt",
            "Socratic Logs": "./memory/logs/socraticlogs.txt"
        }
        
        all_ok = True
        for name, path in files.items():
            exists = os.path.exists(path)
            status = "✓" if exists else "✗"
            print(f"{status} {name} file exists")
            if not exists:
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"✗ Error checking SocraticReasoning: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_autonomous_mode():
    """Verify autonomous mode setup"""
    print("\n" + "="*60)
    print("CHECKING AUTONOMOUS MODE SETUP")
    print("="*60)
    
    try:
        from automind.openmind import OpenMind
        
        openmind = OpenMind()
        print("✓ OpenMind initialized")
        
        # Check if reasoning_loop method exists
        if hasattr(openmind, 'reasoning_loop'):
            print("✓ reasoning_loop method exists")
        else:
            print("✗ reasoning_loop method missing")
            return False
        
        # Check if autonomous_reasoning flag exists
        if hasattr(openmind, 'autonomous_reasoning'):
            print(f"✓ autonomous_reasoning flag exists: {openmind.autonomous_reasoning}")
        else:
            print("✗ autonomous_reasoning flag missing")
            return False
        
        # Check if reasoning_task tracking exists
        if hasattr(openmind, 'reasoning_task'):
            print("✓ reasoning_task tracking exists")
        else:
            print("✗ reasoning_task tracking missing")
            return False
        
        # Check if _create_task method exists
        if hasattr(openmind, '_create_task'):
            print("✓ _create_task method exists")
        else:
            print("✗ _create_task method missing")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error checking autonomous mode: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_writing():
    """Test writing to memory files"""
    print("\n" + "="*60)
    print("TESTING MEMORY WRITING")
    print("="*60)
    
    try:
        from memory.memory import (
            create_memory_folders,
            store_in_stm,
            save_conversation_memory,
            save_internal_reasoning,
            save_valid_truth,
            DialogEntry
        )
        
        # Ensure folders exist
        create_memory_folders()
        print("✓ Memory folders created")
        
        # Test STM storage
        test_entry = DialogEntry("test instruction", "test response")
        store_in_stm(test_entry)
        print("✓ STM storage works")
        
        # Test conversation memory
        test_memory = {"dialog": {"instruction": "test", "response": "test"}}
        save_conversation_memory(test_memory)
        print("✓ Conversation memory storage works")
        
        # Test internal reasoning
        test_reasoning = {
            "timestamp": int(datetime.now().timestamp()),
            "prompt": "test prompt",
            "conclusion": "test conclusion"
        }
        save_internal_reasoning(test_reasoning)
        print("✓ Internal reasoning storage works")
        
        # Test valid truth
        test_truth = {
            "expression": "test expression",
            "timestamp": datetime.now().isoformat()
        }
        save_valid_truth(test_truth)
        print("✓ Valid truth storage works")
        
        return True
    except Exception as e:
        print(f"✗ Error testing memory writing: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_recent_activity():
    """Check for recent activity in log files"""
    print("\n" + "="*60)
    print("CHECKING RECENT ACTIVITY")
    print("="*60)
    
    import time
    
    log_files = {
        "Premises": "./memory/logs/premises.json",
        "Thoughts": "./memory/logs/thoughts.json",
        "Conclusions": "./memory/logs/conclusions.txt",
        "Truth Tables": "./memory/logs/truth.json"
    }
    
    current_time = time.time()
    recent_threshold = 3600  # 1 hour
    
    for name, path in log_files.items():
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            age = current_time - mtime
            age_hours = age / 3600
            
            if age < recent_threshold:
                status = "✓"
                print(f"{status} {name}: Last updated {age_hours:.2f} hours ago")
            else:
                status = "⚠"
                print(f"{status} {name}: Last updated {age_hours:.2f} hours ago (old)")
        else:
            print(f"✗ {name}: File does not exist")

def main():
    """Run all verification checks"""
    print("\n" + "="*60)
    print("LMAGI SYSTEM VERIFICATION")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    results = {
        "Memory Folders": check_memory_folders(),
        "Log Files": check_log_files(),
        "Truth Table Logging": check_truth_table_logging(),
        "Socratic Reasoning": check_socratic_reasoning(),
        "Autonomous Mode": check_autonomous_mode(),
        "Memory Writing": test_memory_writing(),
    }
    
    check_recent_activity()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {check}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
    else:
        print("✗ SOME CHECKS FAILED - Review output above")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

