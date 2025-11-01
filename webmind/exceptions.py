# exceptions.py (c) Gregory L. Magnusson MIT license 2024
# Custom exception classes for standardized error handling

class LmagiException(Exception):
    """Base exception for lmagi application"""
    pass

class APIError(LmagiException):
    """API-related errors"""
    def __init__(self, message: str, api_name: str = None, status_code: int = None):
        super().__init__(message)
        self.api_name = api_name
        self.status_code = status_code

class ReasoningError(LmagiException):
    """Reasoning engine errors"""
    pass

class MemoryError(LmagiException):
    """Memory storage errors"""
    pass

class ConfigurationError(LmagiException):
    """Configuration and setup errors"""
    pass

class ValidationError(LmagiException):
    """Data validation errors"""
    pass

