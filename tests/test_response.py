# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

import pytest

from prympt import Output, Response
from prympt.output import outputs_to_xml

lorem_ipsum = """Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua."""

outputs = [
    Output(description="Sample text"),
    Output(name="python", content='a = "10"'),
    Output(name="text", content=lorem_ipsum),
    Output(name="sh", content="ls -lh\nchmod 755 test.txt"),
    Output(name="text2", description="This is another sample text"),
    Output(name="cpp", content="int main() {}"),
]

response_outputs = "These are some random outputs\n\n" + outputs_to_xml(outputs)


def test_response_constructor() -> None:
    """Test the constructor of the Response class."""
    response = Response(response_outputs)

    assert response.__str__().startswith("These are some random outputs")
    assert str(response) == response_outputs


def test_codeblocks_iteration() -> None:
    """Test the iteration over codeblocks in the Response."""
    response = Response(response_outputs)

    assert len(response) == len(outputs)

    for i in range(len(response)):
        assert response[i] == outputs[i]

    for index, codeblock in enumerate(response):
        assert (
            codeblock == outputs[index]
        ), f"{codeblock} is not equal to {outputs[index]}"


@pytest.mark.parametrize("output", outputs)
def test_response_contains(output: Output) -> None:
    """Test the __contains__ method of the Response class with various substrings."""
    if output.name:
        response = Response(response_outputs)
        assert hasattr(response, output.name)
        assert getattr(response, output.name) == output.content

        assert output.name in response
        assert "Lorem_Ipsum" not in response
