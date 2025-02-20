# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

import ast
import builtins
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, List, Union
from xml.dom import minidom

from .exceptions import MalformedOutput


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
            try:
                parsed_type = getattr(builtins, self.type)
                if parsed_type is not str:
                    self.content = parsed_type(ast.literal_eval(self.content))
            except AttributeError:
                raise MalformedOutput(
                    f"Could not parse suggested type '{self.type}' for parameter '{self.name}' in response"
                )
            except SyntaxError:
                raise MalformedOutput(
                    f"Got value '{self.content}' for parameter '{self.name}' does not convert into type '{self.type}' suggested in response"
                )


def outputs_to_xml(outputs: List[Output]) -> str:
    """
    Convert a list of `Output` objects into an XML string.
    - The fields `type`, `name`, and `description` are XML attributes.
    - The `content` field is the text body of the <output> element.
    """
    root = ET.Element("outputs")  # You can choose any root tag name you like

    for out in outputs:
        output_elem = ET.SubElement(root, "output")

        # Convert type to string if not already
        if out.name is not None:
            output_elem.set("name", out.name)
        if out.description is not None:
            output_elem.set("description", out.description)
        if out.type is not None:
            output_elem.set("type", out.type)

        # content goes as text inside the <output> element
        if out.content is not None:
            output_elem.text = str(out.content)

    # Convert the ElementTree to a string
    xml_str = ET.tostring(root, encoding="unicode").strip()

    # Indent
    dom = minidom.parseString(xml_str)
    pretty_xml_str = str(dom.toprettyxml(indent="  "))  # Use 2 spaces for indentation

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
