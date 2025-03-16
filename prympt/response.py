# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

from __future__ import (  # Required for forward references in older Python versions
    annotations,
)

from typing import Iterator, List, Tuple, Any
import json
from .prompt import Prompt
from .output import Output, xml_to_outputs
from .exceptions import ResponseError, ToolCallError

def tool_call_to_message(id, name, result):
    return {
        "role": "tool",
        "tool_call_id": id,
        "name": name,
        "content": result,
    }
    
def _parse_llm_response(llm_response:Any) -> Tuple[str, Any]:
    
    if not llm_response:
        return "", []
    
    # Assume llm_response is a litellm Message
    try:
        # Get tool calls from message
        tool_calls = []
        for tool_call in llm_response.tool_calls:
            id = tool_call.id
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool_calls.append([id, name, arguments])
        return llm_response.content, tool_calls
    
    except Exception as e:
        pass

    # llm_response must be a string
    assert isinstance(llm_response, str), f"Error, llm_response is of type {type(llm_response)}"
    return llm_response, []
    

class Response:
    """
    A class representing a response containing code blocks.
    """

    def __init__(
        self, llm_response: Any,
        prompt: Prompt = Prompt(),
        tool_calls: List[Tuple[str, Any]] = None,        
        ):
        """
        Initializes a Response object. Parses the response text, according to the prompt content.
        """
        
        response_text, tool_calls = _parse_llm_response(llm_response)
        
        self.__raw_response_text: str = response_text if response_text else ""

        self.__outputs: List[Output] = xml_to_outputs(self.__raw_response_text)
        self.__tool_calls:List[Tuple[str, Any]] = tool_calls
        
        # Add output contents as member variables in response object
        for output in self.__outputs:
            if output.name and not hasattr(self, output.name):
                setattr(self, output.name, output.content)

        # Check return types with prompt outputs
        if prompt.outputs:

            # Check that expected and responded outputs are compatible
            if len(prompt.outputs) != self.__len__():
                raise ResponseError(f"Expected {len(prompt.outputs)} outputs in LLM response, but got {self.__len__()}")

            new_errors = []
            for index, (defined, responded) in enumerate(
                zip(prompt.outputs, self)
            ):
                if defined.name != responded.name:
                    new_errors += [
                        f"Name for output at position {index} ('{defined.name}') differs from the one provided by LLM ('{responded.name}')\n"
                    ]
                if defined.type != responded.type:
                    new_errors += [
                        f"Type for output at position {index} ('{defined.type}') differs from the one provided by LLM ('{responded.type}')\n"
                    ]

            if new_errors:
                raise ResponseError("\n".join(new_errors))

        if self.__tool_calls:
            
            self.tool_calls = []
            # Appending output of function call
            for id, name, arguments in self.__tool_calls:
                
                if name not in prompt.tools:
                    raise ResponseError(f"Unknown tool with name '{name}'")                
                
                try:
                    tool_call_result = prompt.tools[name].callable(**arguments)
                except Exception as e:
                    raise ToolCallError(e.message)
                
                tool_call_message = tool_call_to_message(id, name, tool_call_result)
                
                self.tool_calls.append(tool_call_message)
                
    def __str__(self) -> str:
        """Returns the raw response text."""
        return self.__raw_response_text

    def __iter__(self) -> Iterator[Output]:
        """Returns an iterator over the code blocks."""
        return iter(self.__outputs)

    def __len__(self) -> int:
        """Returns the number of code blocks."""
        return len(self.__outputs)

    def __getitem__(self, index: int) -> Output:
        """Returns the code block at the given index."""
        return self.__outputs[index]

    def __contains__(self, name: str) -> bool:
        """
        Checks if a code block with the given language exists.

        Args:
            name (str): The output name.

        Returns:
            bool: True if output with that name exists in response, False otherwise.
        """
        return any(output.name == name for output in self.__outputs)
