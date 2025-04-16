# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

import inspect
import ast
from typing import get_type_hints
from pydantic import create_model, ValidationError
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import (
    Any,
    Tuple,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
import uuid
from lxml import etree

import docstring_parser
from dataclasses import dataclass
from typing import Callable, Dict, Any

from .exceptions import ToolInitializationError
from .output import find_last_xml_block

@dataclass
class Tool:
    name: str
    func: Callable
    schema: Dict[str, Any]
    signature: str
    
    def __init__(self, _callable:Any) -> None:
        # Assuming '_callable' is a function
        self.name = _callable.__name__
        self.func = _callable
        self.signature = summarize_function(_callable)
        self.schema = function_to_json_schema(_callable)
    
    @property
    def to_xml(self) -> str:
        return tools_to_xml([self])

    def __call__(self, **kwargs):
        return self.func(**validate_and_cast_tool_parameters(self.func, kwargs))

def summarize_function(func: Callable) -> str:
    """
    Returns a one‑line summary of `func` in the form:
      func_name(parameters) – first line of its docstring
    """
    sig = inspect.signature(func)
    doc = (func.__doc__ or "").strip().splitlines()
    summary = doc[0] if doc else "<no docstring>"
    return f"{func.__name__}{sig} - {summary}"

def python_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """
    Maps a Python type to a JSON schema fragment.
    """
    # Basic types
    if py_type is str:
        return {"type": "string"}
    elif py_type is int:
        return {"type": "integer"}
    elif py_type is float:
        return {"type": "number"}
    elif py_type is bool:
        return {"type": "boolean"}

    # List types (e.g., List[T] or list)
    if py_type is list or get_origin(py_type) is list:
        args = get_args(py_type)
        if args:
            return {"type": "array", "items": python_type_to_json_schema(args[0])}
        return {"type": "array"}

    # Dict types (e.g., Dict[K, V] or dict)
    if py_type is dict or get_origin(py_type) is dict:
        return {"type": "object"}

    # Handle Union types, including Optional[T]
    if get_origin(py_type) is Union:
        args = get_args(py_type)
        # Special case for Optional[T] (i.e., Union[T, None])
        if len(args) == 2 and type(None) in args:
            non_none = args[0] if args[1] is type(None) else args[1]
            return python_type_to_json_schema(non_none)
        else:
            # For a union of multiple types, use anyOf
            return {"anyOf": [python_type_to_json_schema(arg) for arg in args]}

    # Fallback for types we do not recognize
    return {"type": "string"}


def json_schema_to_python_type(schema: Dict[str, Any]) -> Any:
    """
    Attempts to map a JSON schema fragment back to a Python type.
    This mapping is simple and covers basic cases.
    """
    schema_type = schema.get("type")
    if schema_type == "string":
        return str
    elif schema_type == "integer":
        return int
    elif schema_type == "number":
        return float
    elif schema_type == "boolean":
        return bool
    elif schema_type == "array":
        items = schema.get("items")
        if items:
            item_type = json_schema_to_python_type(items)
            return List[item_type]  # type: ignore
        return list
    elif schema_type == "object":
        return dict
    elif "anyOf" in schema:
        types = tuple(json_schema_to_python_type(s) for s in schema["anyOf"])
        return Union[types]
    return str


def function_to_json_schema(func: Callable[..., Any]) -> Dict[str, Any]:
    """
    Converts a Python function to a JSON schema for LLM function calling.
    Separates the function description from the parameter descriptions by
    parsing the function's docstring using the docstring_parser library.

    Args:
        func: The Python function to convert.

    Returns:
        A dictionary representing the JSON schema.
    """
    # Parse the docstring using docstring_parser.
    doc = docstring_parser.parse(inspect.getdoc(func) or "")
    # Combine the short and long descriptions for the overall function description.
    doc_description = doc.short_description or ""
    if doc.long_description:
        doc_description += " " + doc.long_description

    # Build a dictionary mapping parameter names to their descriptions.
    param_descriptions = {param.arg_name: param.description for param in doc.params if param.arg_name}

    name = func.__name__
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        annotation = type_hints.get(param_name, str)
        param_schema = python_type_to_json_schema(annotation)
        # Add the parameter description if available.
        if param_name in param_descriptions and param_descriptions[param_name]:
            param_schema["description"] = param_descriptions[param_name]
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        properties[param_name] = param_schema

    schema: Dict[str, Any] = {
        "name": name,
        "description": doc_description.strip(),
        "parameters": {
            "type": "object",
            "properties": properties,
        },
    }
    if required:
        schema["parameters"]["required"] = required

    return schema

def tools_to_xml(tools: List[Tool]) -> str:
    """
    Converts a list of Tool objects into an XML string.
    The parameters for each tool are extracted from the tool's schema.
    Each parameter includes a 'required' field indicating whether it is required.

    Args:
        tools: A list of Tool objects.

    Returns:
        A string representing the XML structure with no indentation.
    """
    # Create the root element
    tool_calls = etree.Element('tool_calls')

    # Iterate over the list of tools
    for tool in tools:
        # Create the tool_call element
        tool_call = etree.SubElement(tool_calls, 'tool_call', name=tool.name)

        # Extract parameters and required fields from the tool's schema
        parameters = tool.schema['parameters']['properties']
        required_params = tool.schema['parameters'].get('required', [])

        # Add parameters to the tool_call element
        for param_name, param_schema in parameters.items():
            param_type = param_schema.get('type', 'str')  # Default to 'str' if type is not specified
            # Check if the parameter is required
            is_required = param_name in required_params
            # Create the param element with the 'required' attribute
            param_element = etree.SubElement(
                tool_call,
                'param',
                name=param_name,
                #type=param_type,
                #required=str(is_required).lower()  # Convert boolean to 'true' or 'false',
            )
            # Add a CDATA section for the parameter value
            param_element.text = etree.CDATA(f"... value for param '{param_name}' goes here ...")

    # Convert the ElementTree to a string with no indentation
    xml_str = etree.tostring(tool_calls, encoding='unicode', pretty_print=True)

    return xml_str

def xml_to_tool_calls(text: str) -> list:
    """
    Parses a string containing a <tool_calls> block and extracts the function names and parameters.
    
    If an id is not provided in the XML for a tool call, a random id is generated.
    
    Args:
        text (str): A string containing the <tool_calls> XML block.

    Returns:
        list: A list of dictionaries, where each dictionary is formatted as:
              {
                  'id': <tool call id (str)>,
                  'type': 'function',
                  'function': {
                      'name': <tool call name (str)>,
                      'arguments': <JSON string of parameters dict>
                  }
              }
    """
    
    # Extract the XML block containing <tool_calls> from the input text.
    xml_string = find_last_xml_block(text, 'tool_calls')
    if not xml_string:
        return []

    # Parse the XML string
    try:
        root = etree.fromstring(xml_string)
    except etree.XMLSyntaxError as e:
        raise ToolInitializationError(f"Error parsing tool calls in XML: {e}")

    result = []

    # Iterate over each <tool_call> element
    for tool_call in root.findall('tool_call'):
        # Use the provided id if available; otherwise, generate a new one.
        tool_id = "call_" + uuid.uuid4().hex
        
        # Extract the tool call name
        tool_name = tool_call.get('name')
        
        # Build the parameters dictionary.
        params = {}
        for param in tool_call.findall('param'):
            param_name = param.get('name')
            # Remove leading/trailing whitespace if text is provided
            param_value = param.text.strip() if param.text else ""
            if param_name and param_value:
                params[param_name] = param_value

        # Construct the output dictionary for this tool call.
        output_entry = {
            'id': tool_id,
            'type': 'function',
            'function': {
                'name': tool_name,
                'arguments': json.dumps(params)
            }
        }
        result.append(output_entry)

    return result


def get_function_signature_from_schema(schema):
    """
    Given a tool schema dictionary following OpenAI's API format,
    returns the function signature as a string.
    
    Expected schema structure:
    
    {
        "name": "function_name",
        "description": "Description of the function",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "int", "description": "desc for param1"},
                "param2": {"type": "str", "description": "desc for param2"},
                ...
            },
            "required": ["param1", ...]
        }
    }
    """
    params = []
    parameters = schema.get("parameters", {})
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    
    for param_name, param_info in properties.items():
        param_type = param_info.get("type", "Any")
        # Build the parameter string with type annotation.
        param_str = f"{param_name}: {param_type}"
        # If the parameter is not required, we add a default value of None.
        if param_name not in required:
            param_str += " = None"
        params.append(param_str)
    
    return f"{schema['name']}({', '.join(params)}): {schema.get('description')}"


def validate_and_cast_tool_parameters(func, params: dict) -> dict:
    """
    Validate and coerce a dictionary of string‑or‑native values against a target function’s signature.

    This utility will:

    1. Reject any keys in `params` that are not actual parameters of `func`.  
    2. Enforce presence of all required parameters (those without default values).  
    3. Automatically parse string literals for built‑in container types (list, dict, tuple, set) via `ast.literal_eval`.  
    4. Build a temporary Pydantic model whose fields mirror `func`’s signature (including defaults) and leverage Pydantic’s powerful coercion & validation.  
    5. Return a fully typed `dict` suitable for passing into `func`.

    ### Design choices

    - **inspect.signature + get_type_hints**: ensures runtime reflection of parameter names, defaults, and annotations.  
    - **ast.literal_eval**: safely converts string representations of containers before handing off to Pydantic.  
    - **Pydantic create_model**: centralizes type coercion/validation (including nested and optional types) with concise error reporting.  
    - **ToolInitializationError**: a single exception type for all validation failures, simplifying caller error handling.

    ### Requirements

    - Python ≥3.8  
    - Pydantic ≥2.0  
    - Importable `ToolInitializationError` for raising validation errors  

    ### Parameters

    - **func** (`callable`): target function whose signature & type hints drive validation.  
    - **params** (`dict[str, Any]`): mapping from parameter name → string or native value.

    ### Returns

    - **dict[str, Any]**: the same keys as `params`, but with values coerced into the types declared on `func`.

    ### Raises

    - **ToolInitializationError** if:
        - Unexpected parameters are present.
        - Required parameters are missing.
        - A container literal fails to parse.
        - A value cannot be coerced into its annotated type.

    ### Example

    ```python
    from typing import Optional, List, Dict

    def fn(a: int, b: Optional[str] = None, c: List[float] = [1.0]):
        ...

    params = {"a": "123", "c": "[2.5, 3.0]"}
    validated = validate_and_cast(fn, params)
    # → {"a": 123, "b": None, "c": [2.5, 3.0]}
    ```
    """
        
    sig   = inspect.signature(func)
    hints = get_type_hints(func)

    # Catch any keys that aren’t actual function parameters
    unexpected = set(params) - set(sig.parameters)
    if unexpected:
        raise ToolInitializationError(f"Unexpected parameter(s): {', '.join(sorted(unexpected))}")

    # Missing‑required check (skip params that have a default)
    required = set(name for name, param in sig.parameters.items() if param.default is inspect._empty)
    missing = required - set(params)
    if missing:
        raise ToolInitializationError(f"Missing required parameter(s): {', '.join(sorted(required))}")

    # Pre‑parse any container literals
    parsed = {}
    for name, raw in params.items():
        expected = hints.get(name, Any)
        origin = getattr(expected, "__origin__", None)
        if isinstance(raw, str) and origin in (list, dict, tuple, set):
            try:
                raw = ast.literal_eval(raw)
            except Exception:
                raise ToolInitializationError(
                    f"Failed to parse value for parameter '{name}'. "
                    f"Expected value of type '{expected}', got {raw!r}"
                )
        parsed[name] = raw

    # Build Pydantic model using defaults where provided
    fields = {}
    for name, param in sig.parameters.items():
        annotation = hints.get(name, Any)
        default = param.default if param.default is not inspect._empty else ...
        fields[name] = (annotation, default)

    model = create_model(func.__name__ + "Params", **fields)

    try:
        return model(**parsed).dict()
    except ValidationError as exc:
        err   = exc.errors()[0]
        param = err["loc"][0]
        expected = hints.get(param, Any)
        got = params.get(param)
        raise ToolInitializationError(
            f"Failed to validate parameter '{param}'. Expected {expected}, got {got!r}"
        )


def test_tools(tools:List[Tool]):

    tool_names = [ tool.name for tool in tools ]
    
    overlapping_tools = [ tool_name for tool_name, count in Counter(tool_names).items() if count > 1]
    if overlapping_tools:
        raise ToolInitializationError(
            f"Tools with overlapping names: {', '.join(overlapping_tools)}"
        )
    
def tools_to_schemas(tools:List[Tool]):

    test_tools(tools)
    return [ { "type": "function", "function": tool.schema } for tool in tools ]

def tools_to_prompt(tools:List[Tool]):

    test_tools(tools)
    
    from .prompt import Prompt

    if not tools:
        return Prompt("")

    # Compose string for signatures
    signatures = []
    for tool in tools:
        signatures += [ tool.signature ]

    signatures = "  - " + "\n  - ".join(signatures)

    # Compose string for sample tool call
    def tool_name(param1_name: str, param2_name: int, param3_name: int):
        """Sample tool"""
        pass
    
    sample_tool_xml = Tool(tool_name).to_xml

    # Combine into tools template
    return Prompt(
        "\n\nThis is a list of the tools available:\n" +
        signatures +
        "\n\nProvide any and all tool cals inside a single XML following this format:\n\n" +
        sample_tool_xml
    )