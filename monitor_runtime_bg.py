#!/usr/bin/env python3
"""
Background monitor - runs app and logs to file
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime
from pathlib import Path

log_file = 'runtime_monitor.log'
monitor_duration = 900  # 15 minutes

def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")

def get_file_size(filepath):
    try:
        return os.path.getsize(filepath) if os.path.exists(filepath) else 0
    except:
        return 0

def check_recent_entries(filepath, max_lines=2):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r') as f:
            if filepath.endswith('.json'):
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data[-max_lines:]
                except:
                    lines = f.readlines()
                    entries = []
                    for line in lines[-max_lines:]:
                        try:
                            entries.append(json.loads(line.strip()))
                        except:
                            pass
                    return entries
            else:
                lines = f.readlines()
                return [line.strip() for line in lines[-max_lines:] if line.strip()]
    except:
        return []

def main():
    # Clear previous log
    with open(log_file, 'w') as f:
        f.write(f"=== LMAGI Runtime Monitor Started at {datetime.now()} ===\n")
    
    log("Starting lmagi backend server...")
    
    venv_python = os.path.join(os.path.dirname(__file__), 'venv', 'bin', 'python')
    lmagi_script = os.path.join(os.path.dirname(__file__), 'lmagi.py')
    
    env = os.environ.copy()
    env['LMAGI_HEADLESS'] = '1'
    
    process = subprocess.Popen(
        [venv_python, lmagi_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1
    )
    
    log(f"Process started (PID: {process.pid})")
    log("Application available at http://localhost:8080")
    log("Monitoring for 15 minutes...\n")
    
    # Track file sizes
    files_to_monitor = [
        'memory/logs/thoughts.json',
        'memory/logs/conclusions.txt',
        'memory/logs/truth.json',
        'memory/logs/premises.json',
        'memory/logs/notpremise.json',
        'memory/truth/logs.txt',
    ]
    
    file_sizes = {f: get_file_size(f) for f in files_to_monitor}
    
    start_time = time.time()
    last_summary = start_time
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = monitor_duration - elapsed
            
            if remaining <= 0:
                log("15 minute timeout reached")
                break
            
            if process.poll() is not None:
                log(f"Process ended (exit code: {process.returncode})")
                break
            
            # Check for app output
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    log(f"APP: {line.strip()}")
            
            # Check for file changes
            for filepath in files_to_monitor:
                current_size = get_file_size(filepath)
                if current_size > file_sizes[filepath]:
                    log(f"FILE UPDATE: {filepath} ({file_sizes[filepath]} -> {current_size} bytes)")
                    file_sizes[filepath] = current_size
                    
                    # Show recent entries
                    recent = check_recent_entries(filepath, 1)
                    if recent:
                        log(f"  Recent entry: {str(recent[-1])[:150]}")
            
            # Summary every 30 seconds
            if current_time - last_summary >= 30:
                log(f"STATUS: Running for {int(elapsed)}s, {int(remaining)}s remaining")
                log(f"  File sizes: {[(f, get_file_size(f)) for f in files_to_monitor]}")
                last_summary = current_time
            
            time.sleep(2)
    
    except KeyboardInterrupt:
        log("Interrupted by user")
    finally:
        log("Shutting down...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
        log("\n=== Final Summary ===")
        for filepath in files_to_monitor:
            size = get_file_size(filepath)
            log(f"  {filepath}: {size} bytes")
        log(f"Full log saved to: {log_file}")

if __name__ == '__main__':
    main()

