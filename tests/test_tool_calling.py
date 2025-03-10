# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

from typing import (
    List,
    Optional,
)

from prympt.tool_call import (
    Parameter,
    ToolCalling,
    function_to_json_schema,
    json_schema_to_tool_calling,
    python_type_to_json_schema,
    tool_calling_to_json_schema,
)


# Sample function for demonstration and testing.
def sample_function(a: int, b: Optional[str] = None, c: List[float] = [1.0]) -> int:
    """
    Sample function docstring.
    """
    return a


def test_function_to_json_schema() -> None:
    schema = function_to_json_schema(sample_function)
    # Check function name and docstring.
    assert schema["name"] == "sample_function"
    assert "Sample function docstring" in schema["description"]

    # Check parameters.
    params = schema["parameters"]
    assert params["type"] == "object"
    properties = params["properties"]
    # Verify each parameter's JSON schema.
    assert properties["a"] == {"type": "integer"}
    # Optional[str] is mapped to string.
    assert properties["b"] == {"type": "string"}
    assert properties["c"] == {"type": "array", "items": {"type": "number"}}
    # Check required parameters (only "a" is required).
    assert "required" in params
    assert params["required"] == ["a"]


def test_tool_calling_to_json_schema() -> None:
    tool = ToolCalling(
        name="sample_function",
        parameters=[
            Parameter(name="a", required=True, type=int),
            Parameter(name="b", required=False, type=str),
            Parameter(name="c", required=False, type=List[float]),
        ],
    )
    schema = tool_calling_to_json_schema(tool)
    # Check tool name.
    assert schema["name"] == "sample_function"
    params = schema["parameters"]
    properties = params["properties"]
    # Check each parameter's JSON schema.
    assert properties["a"] == {"type": "integer"}
    assert properties["b"] == {"type": "string"}
    assert properties["c"] == {"type": "array", "items": {"type": "number"}}
    # Check required fields.
    assert "required" in params
    assert params["required"] == ["a"]


def test_json_schema_to_tool_calling() -> None:
    # Create a ToolCalling instance and convert to JSON schema.
    tool = ToolCalling(
        name="sample_function",
        parameters=[
            Parameter(name="a", required=True, type=int),
            Parameter(name="b", required=False, type=str),
            Parameter(name="c", required=False, type=List[float]),
        ],
    )
    schema = tool_calling_to_json_schema(tool)
    # Convert JSON schema back to a ToolCalling instance.
    restored_tool = json_schema_to_tool_calling(schema)
    assert restored_tool.name == tool.name
    # Verify that each parameter is restored correctly.
    assert len(restored_tool.parameters) == len(tool.parameters)
    for orig_param in tool.parameters:
        matching = [p for p in restored_tool.parameters if p.name == orig_param.name]
        assert matching, f"Parameter {orig_param.name} not found in restored tool."
        restored_param = matching[0]
        assert restored_param.required == orig_param.required
        # Compare types by converting them to JSON schema fragments.
        orig_type_schema = python_type_to_json_schema(orig_param.type)
        restored_type_schema = python_type_to_json_schema(restored_param.type)
        assert orig_type_schema == restored_type_schema


def test_chain_conversion() -> None:
    """
    Test the full chain:
    sample_function -> JSON schema -> ToolCalling -> JSON schema.
    """
    # Convert sample_function to JSON schema.
    schema_from_function = function_to_json_schema(sample_function)
    # Convert JSON schema to ToolCalling.
    tool_from_schema = json_schema_to_tool_calling(schema_from_function)
    # Convert ToolCalling back to JSON schema.
    schema_from_tool = tool_calling_to_json_schema(tool_from_schema)

    # Check that the name remains the same.
    assert schema_from_tool["name"] == schema_from_function["name"]

    # Compare parameters (description may differ).
    props1 = schema_from_function["parameters"]["properties"]
    props2 = schema_from_tool["parameters"]["properties"]
    assert props1 == props2

    # Check that the required list is preserved.
    req1 = sorted(schema_from_function["parameters"].get("required", []))
    req2 = sorted(schema_from_tool["parameters"].get("required", []))
    assert req1 == req2
