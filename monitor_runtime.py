#!/usr/bin/env python3
"""
Monitor lmagi application and watch logs in real-time
Runs for 5 minutes with timeout
"""

import os
import sys
import time
import subprocess
import signal
import json
from datetime import datetime
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")

def print_log(tag, message, color=Colors.GREEN):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{timestamp}] {tag}: {message}{Colors.RESET}")

def monitor_log_file(filepath, last_position, label):
    """Monitor a log file for new entries"""
    if not os.path.exists(filepath):
        return last_position
    
    try:
        with open(filepath, 'r') as f:
            f.seek(last_position)
            new_content = f.read()
            if new_content:
                lines = new_content.strip().split('\n')
                for line in lines:
                    if line.strip():
                        print_log(label, line, Colors.YELLOW)
                return f.tell()
    except Exception as e:
        print_log(f"ERROR-{label}", str(e), Colors.RED)
    
    return last_position

def get_file_size(filepath):
    """Get file size or 0 if doesn't exist"""
    try:
        return os.path.getsize(filepath) if os.path.exists(filepath) else 0
    except:
        return 0

def check_recent_entries(filepath, max_lines=3):
    """Check recent entries in a JSON log file"""
    if not os.path.exists(filepath):
        return []
    
    try:
        with open(filepath, 'r') as f:
            if filepath.endswith('.json'):
                # Try to read as JSON array
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data[-max_lines:]
                except:
                    # Might be line-delimited JSON
                    lines = f.readlines()
                    entries = []
                    for line in lines[-max_lines:]:
                        try:
                            entries.append(json.loads(line.strip()))
                        except:
                            pass
                    return entries
            else:
                # Text file
                lines = f.readlines()
                return [line.strip() for line in lines[-max_lines:] if line.strip()]
    except Exception as e:
        print_log("ERROR", f"Error reading {filepath}: {e}", Colors.RED)
        return []

def main():
    print_header("LMAGI Runtime Monitor")
    print(f"{Colors.BOLD}Starting application and monitoring logs for 15 minutes...{Colors.RESET}\n")
    
    # Track file positions
    file_positions = {
        'memory/logs/thoughts.json': 0,
        'memory/logs/conclusions.txt': 0,
        'memory/logs/truth.json': 0,
        'memory/logs/premises.json': 0,
        'memory/logs/notpremise.json': 0,
        'memory/truth/logs.txt': 0,
    }
    
    # Track file sizes
    file_sizes = {f: get_file_size(f) for f in file_positions.keys()}
    
    # Start the application
    venv_python = os.path.join(os.path.dirname(__file__), 'venv', 'bin', 'python')
    lmagi_script = os.path.join(os.path.dirname(__file__), 'lmagi.py')
    
    env = os.environ.copy()
    env['LMAGI_HEADLESS'] = '1'
    
    print_log("STARTUP", "Starting lmagi backend server...", Colors.BLUE)
    
    try:
        process = subprocess.Popen(
            [venv_python, lmagi_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1
        )
    except Exception as e:
        print_log("ERROR", f"Failed to start application: {e}", Colors.RED)
        return 1
    
    print_log("STARTUP", f"Process started (PID: {process.pid})", Colors.GREEN)
    print_log("INFO", "Application should be available at http://localhost:8080", Colors.CYAN)
    print_log("INFO", "Use the app and watch logs appear here...\n", Colors.CYAN)
    
    start_time = time.time()
    timeout = 900  # 15 minutes
    check_interval = 2  # Check every 2 seconds
    last_summary = start_time
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = timeout - elapsed
            
            if remaining <= 0:
                print_log("TIMEOUT", "15 minute timeout reached", Colors.YELLOW)
                break
            
            # Check if process is still running
            if process.poll() is not None:
                print_log("ERROR", f"Process ended unexpectedly (exit code: {process.returncode})", Colors.RED)
                break
            
            # Monitor stdout/stderr
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    print_log("APP", line.strip(), Colors.CYAN)
            
            # Monitor log files
            for filepath, last_pos in file_positions.items():
                if os.path.exists(filepath):
                    current_size = get_file_size(filepath)
                    if current_size > file_sizes[filepath]:
                        # File has grown, read new content
                        new_pos = monitor_log_file(filepath, last_pos, os.path.basename(filepath))
                        file_positions[filepath] = new_pos
                        file_sizes[filepath] = current_size
            
            # Print summary every 30 seconds
            if current_time - last_summary >= 30:
                print_log("SUMMARY", f"Running for {int(elapsed)}s, {int(remaining)}s remaining", Colors.BLUE)
                
                # Show recent activity
                print_log("ACTIVITY", "Recent entries:", Colors.BLUE)
                for filepath in ['memory/logs/thoughts.json', 'memory/logs/conclusions.txt']:
                    if os.path.exists(filepath):
                        recent = check_recent_entries(filepath, 1)
                        if recent:
                            print_log(f"  {os.path.basename(filepath)}", str(recent[-1])[:100], Colors.YELLOW)
                
                last_summary = current_time
            
            time.sleep(check_interval)
    
    except KeyboardInterrupt:
        print_log("INFO", "Interrupted by user", Colors.YELLOW)
    finally:
        print_log("SHUTDOWN", "Terminating application...", Colors.RED)
        process.terminate()
        
        # Wait for graceful shutdown
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print_log("WARNING", "Force killing process...", Colors.RED)
            process.kill()
            process.wait()
        
        print_header("Final Summary")
        print_log("INFO", "Final log file status:", Colors.BLUE)
        for filepath in file_positions.keys():
            if os.path.exists(filepath):
                size = get_file_size(filepath)
                print(f"  {filepath}: {size} bytes")
        
        print_log("INFO", "Monitoring complete", Colors.GREEN)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

