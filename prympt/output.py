# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

import ast
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, List, Union
from xml.dom import minidom

from lxml import etree

from .exceptions import MalformedOutput, ResponseError


def convert_to_Python_type(value_str: str, type_str: str) -> Any:
    safe_globals = {
        "__builtins__": None,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }

    if type_str not in safe_globals:
        raise ResponseError(
            f"Tried to create Output with a non-standard basic Python type: '{type_str}'"
        )

    try:
        parsed_type = eval(type_str, safe_globals)
        return parsed_type(ast.literal_eval(value_str))
    except SyntaxError:
        raise MalformedOutput(
            f"Could not cast parameter value '{value_str}' to suggested type '{type_str}'"
        )


@dataclass
class Output:
    """A dataclass representing one output of an LLM query."""

    name: Union[str, None] = None
    description: Union[str, None] = None
    content: Union[str, None] = None
    type: Union[str, None] = None

    def __post_init__(self) -> None:

        if self.name and not self.name.isidentifier():
            raise MalformedOutput(
                f"Invalid output name: '{self.name}'. Must comply with Python identifier format: [a-z_][a-z0-9_-]*"
            )

        if self.type and self.content:
            if self.type != "str":
                self.content = convert_to_Python_type(self.content, self.type)


def outputs_to_xml(outputs: List[Output]) -> str:
    """
    Convert a list of Output objects into an XML string.
    - The fields type, name, and description are XML attributes.
    - The content field is wrapped inside a CDATA section within the <output> element.
    """
    root = etree.Element("outputs")  # Root element

    for out in outputs:
        output_elem = etree.SubElement(root, "output")

        # Set attributes
        if out.name is not None:
            output_elem.set("name", out.name)
        if out.description is not None:
            output_elem.set("description", out.description)
        if out.type is not None:
            output_elem.set("type", out.type)

        # Wrap content in a CDATA section
        if out.content is not None:
            output_elem.text = etree.CDATA(str(out.content))

    # Convert the ElementTree to a string
    xml_str = etree.tostring(root, encoding="unicode").strip()

    # Indent using minidom
    dom = minidom.parseString(xml_str)
    pretty_xml_str = dom.toprettyxml(indent="  ")  # Use 2 spaces for indentation

    # Remove the XML declaration line if present:
    lines = pretty_xml_str.splitlines()
    lines = [line for line in lines if not line.strip().startswith("<?xml")]
    pretty_xml_str = "\n".join(lines)

    return pretty_xml_str


def xml_to_outputs(text: str) -> List[Output]:
    """
    Convert an XML string back into a list of `Output` objects.
    - Reads the `type`, `name`, and `description` from attributes.
    - Uses the text content for `content`.
    """

    # Extracts the XML portion from a larger text using a regular expression.
    # Assumes the XML is enclosed in <Outputs>...</Outputs> tags.
    # Use a regular expression to find the XML portion

    matches = list(re.finditer(r"<outputs>.*?</outputs>", text, re.DOTALL))
    if not matches:
        return []
    xml_string = matches[-1].group(0)

    root = ET.fromstring(xml_string)
    output_list = []

    for elem in root.findall("output"):
        # Get attributes or fallback to None
        name = elem.get("name")
        description = elem.get("description")
        out_type = elem.get("type")

        content = elem.text or None  # if there's no text, store None

        output_list.append(
            Output(
                type=out_type,
                content=content,
                name=name,
                description=description,
            )
        )

    return output_list
