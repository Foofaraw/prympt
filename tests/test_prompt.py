# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

import pytest

from prympt import ConcatenationError, Output, Prompt, PromptError, ReplacementError
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


def test_incorrect_Jinja2_syntax() -> None:
    """Test single variable replacement in the prompt template with incorrect Jinja2 syntax."""
    prompt = Prompt("This is a {{ prompt")
    assert prompt.template == "This is a {{ prompt"

    with pytest.warns(
        RuntimeWarning,
        match="unexpected end of template, expected 'end of print statement'",
    ):
        prompt.__str__()


def test_multiple_replacement() -> None:
    """Test multiple variable replacements in the prompt template."""
    prompt = Prompt("This is a {{ prompt }}, so is {{ this }}")
    assert prompt.template == "This is a {{ prompt }}, so is {{ this }}"
    assert (
        prompt(prompt="Prompt", this="This").template == "This is a Prompt, so is This"
    )


def test_multiple_replacement_positional_and_keyword_arguments() -> None:
    """Test multiple variable replacements in the prompt template with positional and keyword arguments."""
    prompt = Prompt("This is a {{ prompt }}, so is {{ this }}")
    assert prompt.template == "This is a {{ prompt }}, so is {{ this }}"
    assert prompt("Prompt", this="This").template == "This is a Prompt, so is This"


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
    assert prompt.get_variables() == ["prompt", "this", "items"]


def test_single_result() -> None:
    """Test adding a single return value to the prompt."""
    prompt = Prompt("Indicate the color of the sky")
    assert prompt.__str__() == "Indicate the color of the sky"

    prompt_returns = prompt.output("text", "color, e.g. red")
    prompt_returns2 = Prompt(
        "Indicate the color of the sky", outputs=[Output("text", "color, e.g. red")]
    )
    assert prompt_returns.__str__() == prompt_returns2.__str__()


def test_multiple_results() -> None:
    """Test adding multiple return values to the prompt."""
    prompt1 = (
        Prompt("Suggest some code and json data")
        .output(name="python", description="python code")
        .output(name="json", description="json code")
    )

    expected_outputs = [
        Output(name="python", description="python code"),
        Output(name="json", description="json code"),
    ]

    obtained_outputs = xml_to_outputs(prompt1.__str__())

    # Query method modifies content to indicate LLM where to put output values
    for expected, obtained in zip(expected_outputs, obtained_outputs):
        assert vars(expected).keys() == vars(obtained).keys()

        for attr in vars(expected):
            if attr != "content":
                assert (
                    vars(expected)[attr] == vars(obtained)[attr]
                ), f"Attribute '{attr}' in expected and obtained object differs: '{vars(expected)[attr]}', '{vars(obtained)[attr]}'"
            else:
                assert (
                    vars(expected)[attr] != vars(obtained)[attr]
                ), f"Attribute 'content' in expected and obtained object is the same: '{vars(expected)[attr]}'"

    prompt2 = Prompt("Suggest some code and json data", outputs=expected_outputs)
    assert prompt1.__str__() == prompt2.__str__()


def test_add_prompts() -> None:
    """Test adding two prompts together."""
    prompt1 = Prompt("Indicate the color of the sky").output(
        name="text", description="color, e.g. red"
    )
    prompt2 = (
        Prompt("Suggest some code and json data")
        .output(name="python", description="python code goes here")
        .output(name="json", content='["a sample string"]')
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
    prompt = Prompt("Indicate the color of the sky").output("text", "color, e.g. red")
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
    prompt1 = Prompt("Indicate the color of the sky").output("text", "color, e.g. red")
    with pytest.raises(PromptError):
        prompt1 += None


def test_replace_keep_outputs() -> None:
    """Test the extraction of variables from the prompt template."""

    prompt1 = Prompt("Test prompt {{var}}").output("var1").output("var2")

    prompt2 = prompt1(var="test")

    assert prompt1.outputs == prompt2.outputs


def test_replace_errors() -> None:
    """Test the extraction of variables from the prompt template."""

    prompt = Prompt("Test prompt {{var1}} {{var2}} {{var3}}")
    with pytest.raises(
        ReplacementError,
        match="Provided 4 positional arguments, but prompt template has only 3 variables",
    ):
        prompt("value1", "value2", "value3", "value4")

    with pytest.raises(
        ReplacementError, match="Got multiple values for template variable 'var1'"
    ):
        prompt("value1", "value2", var1="value1")


def test_str_does_not_mutate_prompt() -> None:

    prompt1 = Prompt("Indicate the color of the sky").output(
        name="text", description="color, e.g. red"
    )
    prompt2 = (
        Prompt("Suggest some code and json data")
        .output(name="python", description="python code goes here")
        .output(name="json", content='["a sample string"]')
    )

    prompt = prompt1 + prompt2

    expected_outputs = [
        Output(name="text", description="color, e.g. red"),
        Output(name="python", description="python code goes here"),
        Output(name="json", content='["a sample string"]'),
    ]

    assert (
        prompt.template
        == "Indicate the color of the sky\nSuggest some code and json data"
    )
    assert prompt.outputs == expected_outputs

    prompt.__str__()

    assert (
        prompt.template
        == "Indicate the color of the sky\nSuggest some code and json data"
    )
    assert prompt.outputs == expected_outputs


def test_check_duplicate_names() -> None:

    prompt1 = Prompt("Indicate the color of the sky").output(
        name="text", description="color, e.g. red"
    )
    prompt2 = (
        Prompt("Suggest some code and json data")
        .output(name="json", content='["a sample string"]')
        .output(name="text", description="code goes here")
    )

    with pytest.raises(
        PromptError, match="Found outputs at positions 0, 2 with same name: 'text'"
    ):
        prompt1 + prompt2
