# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

from __future__ import (  # Required for forward references in older Python versions
    annotations,
)

from typing import Iterator, List, Tuple, Any
import json
from .prompt import Prompt
from .output import Output, xml_to_outputs
from .tool import Tool, xml_to_tool_calls
from .exceptions import ResponseError, ToolCallError


def _llm_response_to_message(llm_response:Any) -> Tuple[str, Any]:
    
    # Check if the response is empty
    if not llm_response:
        return dict( role = 'assistant', content = '')

    # Check if the response is a message
    try:
        message = llm_response.to_dict()
    except AttributeError:
        try:
            assert isinstance(llm_response, dict)
            message = llm_response
        except Exception:
            assert isinstance(llm_response, str)
            message = dict( role = 'assistant', content = llm_response)
            
    if 'tool_calls' not in message or not message['tool_calls']:

        assert message['content']

        if tool_calls := xml_to_tool_calls(message['content']):
            message['tool_calls'] = tool_calls
            
    return message

class Response:
    """
    A class representing a response containing code blocks.
    """

    def set_attribute(self, name:str, content:Any):
        if hasattr(self, name):
            raise ResponseError(f"Tried to add two outputs with name {name}")
        setattr(self, name, content)
        
    def __init__(
        self, llm_response: Any,
        prompt: Prompt = Prompt(),
        tools: List[Tool] = None
        ):
        """
        Initializes a Response object. Parses the response text, according to the prompt content.
        """
                        
        message = _llm_response_to_message(llm_response)                
               
        self.messages = [ message ]
                   
        self.__tools = tools   
        self.__raw_response_text: str = message['content'] if message['content'] else ""

        self.__outputs: List[Output] = xml_to_outputs(self.__raw_response_text)

        self.tool_calls = []

        if message['tool_calls']:
                
            for tool_call in message['tool_calls']:
                id = tool_call['id']
                name = tool_call['function']['name']
                arguments = json.loads(tool_call['function']['arguments'])
                
                tentative_tools = [ tool for tool in tools if tool.name == name ]
                
                if not tentative_tools:
                    raise ToolCallError(f"Tried to use unknown tool with name '{name}'")
                
                if len(tentative_tools) > 1:
                    raise ToolCallError(f"Multiple tools with same name: '{name}'")                
                
                tool = tentative_tools[0]
                
                try:

                    content = tool.func(**arguments)
                    
                    tool_message = dict(
                        role = 'tool',
                        tool_call_id = id,
                        content = content
                    )

                    tool_call = dict(
                        name = name,
                        arguments = arguments,
                        content = content,
                    )

                    self.messages.append(tool_message)
                    
                    self.tool_calls.append(tool_call)
                    
                except Exception as e:
                    raise ToolCallError(f"Using tool '{name}': {str(e)}")
            
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
            
        # Add output contents as member variables in response object
        for output in self.__outputs:
            if output.name:
                self.set_attribute(output.name, output.content)

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
