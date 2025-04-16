# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

from typing import Dict, List

class PrymptError(Exception):
    """Base exception class for Prympt errors."""
    pass

class ToolInitializationError(PrymptError):
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

class ResponseError(PrymptError):
    """Base exception class for response-related errors."""
    
    def __init__(self, description: str, messages: Dict):
        super().__init__(description)
        self.messages = messages
        
class QueryError(PrymptError):
    """Base exception class for query errors."""
    
    def __init__(self, prompt, errors: List[ResponseError]):
        super().__init__(f"Failed LLM query (tried {len(errors)} times)")
        self.prompt = prompt
        self.errors = errors

class OutputError(PrymptError):
    """Base exception class for outputs-related errors."""
    pass
