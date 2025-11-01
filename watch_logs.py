#!/usr/bin/env python3
"""Monitor logs while GUI is running"""
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

def get_size(filepath):
    try:
        return os.path.getsize(filepath) if os.path.exists(filepath) else 0
    except:
        return 0

def show_recent(filepath, max_chars=150):
    if not os.path.exists(filepath):
        return ""
    try:
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        return str(data[-1])[:max_chars]
                except:
                    # Line-delimited JSON
                    lines = f.readlines()
                    if lines:
                        return lines[-1].strip()[:max_chars]
        else:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                if lines:
                    return lines[-1].strip()[:max_chars]
    except:
        pass
    return ""

def main():
    files = [
        'memory/logs/thoughts.json',
        'memory/logs/conclusions.txt',
        'memory/logs/truth.json',
        'memory/logs/premises.json',
        'memory/logs/notpremise.json',
        'memory/truth/logs.txt',
    ]
    
    print("\n" + "="*70)
    print("LMAGI Log Monitor - Watching for 5 minutes")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("GUI application should be running")
    print("="*70 + "\n")
    
    sizes = {f: get_size(f) for f in files}
    start_time = time.time()
    timeout = 300  # 5 minutes
    check_count = 0
    
    try:
        while time.time() - start_time < timeout:
            elapsed = int(time.time() - start_time)
            remaining = int(timeout - elapsed)
            
            if check_count % 3 == 0:  # Every 30 seconds (3 checks * 10s)
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status: {elapsed}s elapsed, {remaining}s remaining")
            
            # Check for changes
            for filepath in files:
                current_size = get_size(filepath)
                if current_size > sizes[filepath]:
                    print(f"\n✓ {filepath} UPDATED")
                    print(f"  Size: {sizes[filepath]} -> {current_size} bytes (+{current_size - sizes[filepath]})")
                    recent = show_recent(filepath)
                    if recent:
                        print(f"  Recent: {recent}")
                    sizes[filepath] = current_size
            
            # Check if app is running
            if check_count % 3 == 0:
                import subprocess
                if subprocess.run(['pgrep', '-f', 'lmagi.py'], capture_output=True).returncode == 0:
                    print("  ✓ Application running")
                else:
                    print("  ✗ Application stopped!")
                    break
            
            check_count += 1
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    print("\n" + "="*70)
    print("Final Summary")
    print("="*70)
    for filepath in files:
        size = get_size(filepath)
        print(f"  {filepath}: {size} bytes")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()

