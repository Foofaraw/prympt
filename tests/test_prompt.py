# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

import pytest

from prympt import (
    ConcatenationError,
    Output,
    Prompt,
    PromptError,
)
from prympt.output import xml_to_outputs


def test_constructor() -> None:
    """Test the constructor of the Prompt class."""
    prompt = Prompt("This is a prompt")
    assert str(prompt) == "This is a prompt"


def test_single_replacement() -> None:
    """Test single variable replacement in the prompt template."""
    prompt = Prompt("This is a {{ prompt }}")
    assert prompt.template == "This is a {{ prompt }}"
    assert str(prompt(prompt="prompt")) == "This is a prompt"


def test_multiple_replacement() -> None:
    """Test multiple variable replacements in the prompt template."""
    prompt = Prompt("This is a {{ prompt }}, so is {{ this }}")
    assert prompt.template == "This is a {{ prompt }}, so is {{ this }}"
    assert (
        prompt(prompt="Prompt", this="This").template == "This is a Prompt, so is This"
    )


template_multiple_variables_loop = """
This is a {{ prompt }}, so is {{ this }}.

{% for item in items %}
    <li>{{ item }}</li>
{% endfor %}
"""

rendered_multiple_variables_loop = """
This is a p, so is t.


    <li>a</li>

    <li>b</li>

    <li>c</li>
"""


def test_multiple_variables_loop() -> None:
    """Test rendering with multiple variables and loops."""
    prompt = Prompt(template_multiple_variables_loop)
    with pytest.warns(RuntimeWarning) as record:
        prompt.__str__()

    warning_text = r"Tried to render prompt that still has undefined Jinja2 variables: (items, prompt, this)"
    assert str(record[-1].message) == warning_text

    assert (
        prompt(prompt="p", this="t", items=["a", "b", "c"]).__str__()
        == rendered_multiple_variables_loop
    )


def test_get_variables() -> None:
    """Test the extraction of variables from the prompt template."""
    prompt = Prompt(template_multiple_variables_loop)
    assert prompt.get_variables() == {"prompt", "items", "this"}


def test_single_result() -> None:
    """Test adding a single return value to the prompt."""
    prompt = Prompt("Indicate the color of the sky")
    assert prompt.__str__() == "Indicate the color of the sky"

    prompt_returns = prompt.returns("text", "color, e.g. red")
    prompt_returns2 = Prompt(
        "Indicate the color of the sky", returns=[Output("text", "color, e.g. red")]
    )
    assert prompt_returns.__str__() == prompt_returns2.__str__()


def test_multiple_results() -> None:
    """Test adding multiple return values to the prompt."""
    prompt1 = (
        Prompt("Suggest some code and json data")
        .returns(name="python", content="python code goes here")
        .returns(name="json", content='["a sample string"]')
    )

    expected_outputs = [
        Output(name="python", content="python code goes here"),
        Output(name="json", content='["a sample string"]'),
    ]

    obtained_outputs = xml_to_outputs(prompt1.__str__())

    # Query method modifies content to indicate LLM where to put output values
    for expected, obtained in zip(expected_outputs, obtained_outputs):
        obtained.content = expected.content

    assert obtained_outputs == expected_outputs

    prompt2 = Prompt("Suggest some code and json data", returns=expected_outputs)
    assert prompt1.__str__() == prompt2.__str__()


def test_add_prompts() -> None:
    """Test adding two prompts together."""
    prompt1 = Prompt("Indicate the color of the sky").returns(
        name="text", description="color, e.g. red"
    )
    prompt2 = (
        Prompt("Suggest some code and json data")
        .returns(name="python", description="python code goes here")
        .returns(name="json", content='["a sample string"]')
    )
    expected_outputs = [
        Output(name="text", description="color, e.g. red"),
        Output(name="python", description="python code goes here"),
        Output(name="json", content='["a sample string"]'),
    ]

    obtained_outputs = xml_to_outputs((prompt1 + prompt2).__str__())

    # Query method modifies content to indicate LLM where to put output values
    for expected, obtained in zip(expected_outputs, obtained_outputs):
        obtained.content = expected.content

    assert obtained_outputs == expected_outputs


def test_add_prompt_string() -> None:
    """Test adding a string to a prompt."""
    prompt = Prompt("Indicate the color of the sky").returns("text", "color, e.g. red")
    rendered_add_prompt_string = "Indicate the color of the sky\nDo it nicely, please"
    assert (
        (prompt + "Do it nicely, please")
        .__str__()
        .startswith(rendered_add_prompt_string)
    )

    expected_outputs = [Output("text", "color, e.g. red")]
    obtained_outputs = xml_to_outputs(prompt.__str__())

    # Query method modifies content to indicate LLM where to put output values
    for expected, obtained in zip(expected_outputs, obtained_outputs):
        obtained.content = expected.content

    assert obtained_outputs == expected_outputs


def test_add_prompt_none() -> None:
    """Test adding None to a prompt, expecting an error."""
    prompt1 = Prompt("Indicate the color of the sky").returns("text", "color, e.g. red")
    with pytest.raises(PromptError):
        prompt1 += None


def test_replace_outputs() -> None:
    """Test the extraction of variables from the prompt template."""

    prompt1 = Prompt("Test prompt {{var}}").returns("var1").returns("var2")
    
    prompt2 = prompt1(var = "test")
    
    assert prompt1.outputs == prompt2.outputs
    
    
