# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

import inspect
import json
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


@dataclass
class Parameter:
    name: str
    required: bool
    type: Any
    value: Optional[Any] = None


@dataclass
class ToolCalling:
    name: str
    parameters: List[Parameter] = field(default_factory=list)


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

    Args:
        func: The Python function to convert.

    Returns:
        A dictionary representing the JSON schema.
    """
    name = func.__name__
    description = func.__doc__ or ""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        annotation = type_hints.get(param_name, str)
        param_schema = python_type_to_json_schema(annotation)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        properties[param_name] = param_schema

    schema: Dict[str, Any] = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
        },
    }
    if required:
        schema["parameters"]["required"] = required

    return schema


def tool_calling_to_json_schema(tool: ToolCalling) -> Dict[str, Any]:
    """
    Converts a ToolCalling instance into a JSON schema.
    """
    properties = {}
    required = []

    for param in tool.parameters:
        param_schema = python_type_to_json_schema(param.type)
        properties[param.name] = param_schema
        if param.required:
            required.append(param.name)

    schema = {
        "name": tool.name,
        "description": "",  # No description in ToolCalling; could be extended if needed.
        "parameters": {"type": "object", "properties": properties},
    }
    if required:
        schema["parameters"]["required"] = required # type: ignore
    return schema


def json_schema_to_tool_calling(schema: Dict[str, Any]) -> ToolCalling:
    """
    Converts a JSON schema into a ToolCalling instance.
    """
    name = schema.get("name", "unknown_tool")
    params_schema = schema.get("parameters", {})
    properties = params_schema.get("properties", {})
    required_list = params_schema.get("required", [])

    parameters: List[Parameter] = []
    for param_name, prop_schema in properties.items():
        py_type = json_schema_to_python_type(prop_schema)
        is_required = param_name in required_list
        parameters.append(
            Parameter(name=param_name, required=is_required, type=py_type)
        )

    return ToolCalling(name=name, parameters=parameters)
