# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

class PrymptError(Exception):
    """Base exception class for Prympt errors."""
    pass

class ToolCallError(PrymptError):
    """Base exception class for Prympt errors."""
    pass

class PromptError(PrymptError):
    """Base exception class for prompt-related errors."""
    pass

class ConcatenationError(PromptError):
    """Exception raised for errors in the input prompt."""
    pass

class ReplacementError(PromptError):
    """Exception class for replacement-related errors."""
    pass

class ResponseError(PrymptError):
    """Base exception class for response-related errors."""
    pass

class MalformedOutput(ResponseError):
    """Exception raised for malformed outputs in responses."""
    pass