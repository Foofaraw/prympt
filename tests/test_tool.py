# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

import pytest


from typing import (
    List,
    Dict,
    Optional,
)

from prympt import (
    Prompt,
    ToolCallError,
    ConcatenationError,
)
from prympt.tool import (
    Tool,
    validate_and_cast,
)

def math_function(
    a: int,
    b: Optional[str] = None,
    c: List[float] = [1.0],
    d: Dict[str, int] = {"x": 1},
):
    return a + len(b) + sum(c) + d.get('y', 0)

def test_validate_and_cast_correct() -> None:
    
    params = dict(a="1", b="hello", c="[2.0, 3.5]", d="{'y': '2'}")
    validated_params = validate_and_cast(math_function, params)
    
    # Correct call with all parameters.
    assert validated_params == dict(a=1, b='hello', c=[2.0, 3.5], d={'y': 2})
    assert math_function(**validated_params) == 13.5
    
def test_validate_and_cast_correct_missing_optional() -> None:    
    # Correct call with missing optional parameter 'd'.
    params = dict(a="1", b="hello", c="[2.0, 3.5]")
    validated_params = validate_and_cast(math_function, params)
    assert validated_params == dict(a=1, b='hello', c=[2.0, 3.5], d={'x':1})
    assert math_function(**validated_params) == 11.5

def test_validate_and_cast_incorrect_missing_required() -> None:    
    # Incorrect call with missing required parameter 'a'.
    params = dict(b="hello", c="[2.0, 3.5]")
    with pytest.raises(ToolCallError):    
        validate_and_cast(math_function, params)

def test_validate_and_cast_incorrect_wrong_parameter() -> None:            
    # Incorrect call with additional non-existing parameter 'e'.
    params = dict(a="1", b="hello", c="[2.0, 3.5]", e="10")
    with pytest.raises(ToolCallError):
        validate_and_cast(math_function, params)

def write_file(path: str, content: str) -> str:
    """
    Writes the provided text content into a file

    Args:
        path (str): The path to the file
        content (str): The content to write to the file
    """
    return ""



# Sample function for demonstration and testing.
def sample_function(a: int, b: Optional[str] = None, c: List[float] = [1.0]) -> int:
    """
    Sample function docstring.
    """
    return a + c[0]


def test_tool() -> None:
    tool_sample_function = Tool(sample_function)
    
    assert tool_sample_function.name == 'sample_function'
    assert tool_sample_function.signature == 'sample_function(a: int, b: Optional[str] = None, c: List[float] = [1.0]) -> int - Sample function docstring.'
    
def test_prompt_tool() -> None:
    """
    Test the full chain:
    sample_function -> JSON schema -> ToolCalling -> JSON schema.
    """

    prompt1 = Prompt("This is a test", tools = [Tool(sample_function)])
    prompt2 = Prompt("This is a test").tool(sample_function)
    
    #from pprint import pprint
    #pprint(prompt.tools)
    assert prompt1.tools == prompt2.tools
    
def test_non_overlapping_tools() -> None:
   
    prompt1 = Prompt("This is a prompt", tools = [Tool(sample_function)])
    prompt2 = Prompt("This is another prompt", tools = [Tool(write_file)])    
    
    prompt = prompt1 + prompt2    
    
    assert sorted(prompt.tools.keys()) == ['sample_function', 'write_file']
    
def test_overlapping_tools() -> None:
    """
    Test the full chain:
    sample_function -> JSON schema -> ToolCalling -> JSON schema.
    """
    
    prompt1 = Prompt("This is a prompt", tools = [Tool(sample_function)])
    prompt2 = Prompt("This is another prompt", tools = [Tool(sample_function)])    
       
    with pytest.raises(
        ConcatenationError,
        match="Trying to concatenate two prompts with overlapping tools: sample_function",
    ):
        prompt1 + prompt2
