# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

from typing import Any, Union

import pytest

from prympt import Output, Prompt, ResponseError
from prympt.output import outputs_to_xml

response_3_tries_no_codeblock = "This is not the answer you're looking for"
response_3_tries_valid = "This is the requested Python code:\n\n" + outputs_to_xml(
    [Output("python", 'a = "10"')]
)


def response_3_tries(
    prompt: Union[str, None] = None, temperature: Union[float, None] = None
) -> str:

    response_3_tries.counter += 1  # type: ignore[attr-defined]

    # print(response_3_tries.counter, temperature)  # type: ignore[attr-defined]

    assert (
        temperature == 0.0 if response_3_tries.counter == 1 else 1.0  # type: ignore[attr-defined]
    ), f"Try {response_3_tries.counter} uses temperature {temperature}"  # type: ignore[attr-defined]

    if response_3_tries.counter < 3:  # type: ignore[attr-defined]
        return response_3_tries_no_codeblock
    else:
        return response_3_tries_valid


def test_3_retries() -> None:

    query_params = dict(
        llm_completion=response_3_tries,
        temperature=0.0,
    )

    prompt_3_tries = Prompt(
        "Generate python code that initializes variable 'a' to 0"
    ).returns("python", "code goes here")

    for retries in range(1, 3):
        response_3_tries.counter = 0  # type: ignore[attr-defined]
        with pytest.raises(ResponseError):
            prompt_3_tries.query(retries=retries, **query_params)

    response_3_tries.counter = 0  # type: ignore[attr-defined]
    response = prompt_3_tries.query(retries=3, **query_params)

    assert response.__str__() == response_3_tries_valid


prompt_wrong_type = """
This is a response with the wrong type:

<outputs>
  <output name="answer" description="The answer to everything" type="float">42.0</output>
</outputs>

"""


def response_wrong_type(prompt: Union[str, None]) -> str:
    return prompt_wrong_type


def test_wrong_type() -> None:

    prompt = Prompt("Answer to everything").returns("anwser", type="int")

    with pytest.raises(ResponseError):
        prompt.query(llm_completion=response_wrong_type, retries=1)


def test_wrong_name() -> None:

    prompt = Prompt("Answer to everything").returns("anser", type="float")

    with pytest.raises(ResponseError):
        prompt.query(llm_completion=response_wrong_type, retries=1)


prompt_incorrect_outputs_number = """
This is a response with the wrong type:

<outputs>
  <output name="answer" description="The answer to everything" type="float">42.0</output>
</outputs>

"""


def response_incorrect_outputs_number(prompt: Union[str, None]) -> str:
    return prompt_wrong_type


def test_incorrect_outputs_number() -> None:

    prompt = Prompt("Answer to everything").returns("anser", type="int").returns("")

    with pytest.raises(ResponseError):
        prompt.query(llm_completion=response_wrong_type, retries=1)
