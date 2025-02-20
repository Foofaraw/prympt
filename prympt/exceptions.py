# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).


class PromptError(Exception):
    """Base exception class for prompt-related errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConcatenationError(PromptError):
    """Exception raised for errors in the input prompt."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ReplacementError(PromptError):
    """Exception class for replacement-related errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ResponseError(PromptError):
    """Base exception class for response-related errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class MalformedOutput(ResponseError):
    """Exception raised for malformed outputs in responses."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
