# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

import pytest

from prympt import MalformedOutput, Output
from prympt.output import outputs_to_xml, xml_to_outputs

response_xml = """
This is a sample response, containing some outputs in the XML

<outputs>
  <output name="greeting" description="A greeting">Hello World</output>
  <output name="answer" description="The answer to everything" type="int">42</output>
  <output name="numbers" description="A list of numbers">[1, 2, 3]</output>
</outputs>

"""


def test_conversion_xml_outputs() -> None:

    # Create a few Output objects
    outputs = [
        Output(content="Hello World", name="greeting", description="A greeting"),
        Output(
            content="42",
            name="answer",
            description="The answer to everything",
            type="int",
        ),
        Output(content="[1, 2, 3]", name="numbers", description="A list of numbers"),
    ]

    xml_string = outputs_to_xml(outputs)

    assert outputs == xml_to_outputs(xml_string)


def test_outputs_extraction() -> None:

    # Create a few Output objects
    outputs = [
        Output(content="Hello World", name="greeting", description="A greeting"),
        Output(
            content="42",
            name="answer",
            description="The answer to everything",
            type="int",
        ),
        Output(content="[1, 2, 3]", name="numbers", description="A list of numbers"),
    ]

    assert outputs == xml_to_outputs(response_xml)


def test_empty_xml() -> None:

    assert xml_to_outputs("This string does not contain an xml") == []


def test_invalid_result_name() -> None:
    with pytest.raises(MalformedOutput):
        Output(name="python code")


def test_invalid_type() -> None:
    with pytest.raises(MalformedOutput):
        Output(
            content="Hello World",
            name="greeting",
            description="A greeting",
            type="message",
        )


def test_wrong_type() -> None:
    with pytest.raises(MalformedOutput):
        Output(
            content="Hello World", name="greeting", description="A greeting", type="int"
        )


response_multiple_xmls = """

<outputs>
  <output name="color" description="Red"/>
</outputs>

However, it's important to note that the perception of warmth in colors can be subjective and can vary based on cultural and personal preferences. In general, red is often considered the warmest color due to its association with fire and heat. Other warm colors include yellow and orange.
```xml
<outputs>
  <output name="color">red</output>
</outputs>
```
"""


def test_multiple_xmls() -> None:

    # Create a few Output objects
    outputs = [Output(name="color", content="red")]

    assert outputs == xml_to_outputs(response_multiple_xmls)
