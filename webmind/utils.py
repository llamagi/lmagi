# utils.py (c) Gregory L. Magnusson MIT license 2024
# Utility functions for error handling and retries

import asyncio
import logging
from functools import wraps
from typing import Callable, TypeVar, Any

T = TypeVar('T')

def retry_with_timeout(max_retries: int = 3, timeout: float = 30.0, backoff: float = 1.0):
    """
    Decorator for async functions to add retry logic with timeout.
    
    Args:
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds for each attempt
        backoff: Backoff multiplier between retries
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                except (asyncio.TimeoutError, Exception) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff * (attempt + 1)
                        logging.warning(
                            f"{func.__name__} failed on attempt {attempt + 1}/{max_retries}: {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logging.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
            # If all retries failed, raise the last exception
            if last_exception:
                raise last_exception
            return None
        return wrapper
    return decorator

def safe_json_load(file_path: str, default: Any = None):
    """
    Safely load JSON from a file with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Default value to return on error (default: empty list)
    
    Returns:
        Loaded JSON data or default value
    """
    import ujson as json
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading JSON from {file_path}: {e}")
        return default if default is not None else []

def safe_json_dump(data: Any, file_path: str) -> bool:
    """
    Safely write JSON to a file with error handling.
    
    Args:
        data: Data to serialize to JSON
        file_path: Path to output file
    
    Returns:
        True if successful, False otherwise
    """
    import ujson as json
    import os
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Error writing JSON to {file_path}: {e}")
        return False

