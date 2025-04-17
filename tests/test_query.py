# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

from typing import Any, Union, Dict, List

import pytest

from prympt import Output, Prompt, ResponseError, QueryError
from prympt.output import outputs_to_xml

response_3_tries_no_codeblock = "This is not the answer you're looking for"
response_3_tries_valid = "This is the requested Python code:\n\n" + outputs_to_xml(
    [Output("python", 'a = "10"')]
)


def response_3_tries(
    prompt: Union[str, None] = None,
    tools: List[Dict] = [],
    temperature: Union[float, None] = None
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


prompt_wrong_type = """
This is a response with the wrong type:

<outputs>
  <output name="answer" description="The answer to everything" type="float">42.0</output>
</outputs>

"""

def response_wrong_type(
    prompt: Union[str, None],
    tools: List[Dict] = [],    
    ) -> str:
    return prompt_wrong_type


prompt_incorrect_outputs_number = """
This is a response with the wrong type:

<outputs>
  <output name="answer" description="The answer to everything" type="float">42.0</output>
</outputs>

"""

def response_incorrect_outputs_number(
    prompt: Union[str, None],
    tools: List[Dict] = [],    
    ) -> str:
    return prompt_wrong_type


def test_3_retries() -> None:

    query_params = dict(
        llm_completion=response_3_tries,
        temperature=0.0,
    )

    prompt_3_tries = Prompt(
        "Generate python code that initializes variable 'a' to 0"
    ).returns("python", "code goes here")

    response_3_tries.counter = 0  # type: ignore[attr-defined]
    with pytest.raises(QueryError) as exc_info:
        prompt_3_tries.query(max_retries=1, **query_params)

    # Check that the error we got is the one expected
    query_error = exc_info.value
    assert query_error.__str__().startswith("Failed LLM query (tried 1 times)")
    assert len(query_error.errors) == 1
    assert query_error.errors[0].__str__() == "Expected 1 outputs in LLM response, but got 0"
    
    for retries in range(2, 3):
        response_3_tries.counter = 0  # type: ignore[attr-defined]
        with pytest.raises(QueryError) as exc_info:
            prompt_3_tries.query(max_retries=retries, **query_params)

        # Check that the error we got is the one expected
        query_error = exc_info.value
        assert query_error.__str__().startswith(f"Failed LLM query (tried {retries} times)")
        assert len(query_error.errors) == retries
        assert query_error.errors[0].__str__() == "Expected 1 outputs in LLM response, but got 0"

    response_3_tries.counter = 0  # type: ignore[attr-defined]
    response = prompt_3_tries.query(max_retries=3, **query_params)

    assert response.__str__() == response_3_tries_valid
    # Check that the errors we got is the one expected
    errors = response.errors    
    assert errors
    assert len(errors) == 2
    assert errors[0].__str__() == "Expected 1 outputs in LLM response, but got 0"


def test_wrong_type() -> None:

    prompt = Prompt("Answer to everything").returns("anwser", type="int")

    with pytest.raises(QueryError) as exc_info:
        prompt.query(llm_completion=response_wrong_type, max_retries=1)

    # Check that the error(s) we got are the one(s) we expected
    query_error = exc_info.value
    assert query_error.__str__().startswith(f"Failed LLM query (tried 1 times)")
    assert len(query_error.errors) == 1
    assert query_error.errors[0].__str__() == "Name for output at position 0 ('anwser') differs from the one provided by LLM ('answer')\n\nType for output at position 0 ('int') differs from the one provided by LLM ('float')\n"

def test_wrong_name() -> None:

    prompt = Prompt("Answer to everything").returns("anser", type="float")

    with pytest.raises(QueryError) as exc_info:
        prompt.query(llm_completion=response_wrong_type, max_retries=1)

    # Check that the error(s) we got are the one(s) we expected
    query_error = exc_info.value
    assert query_error.__str__().startswith(f"Failed LLM query (tried 1 times)")
    assert len(query_error.errors) == 1
    assert query_error.errors[0].__str__() == "Name for output at position 0 ('anser') differs from the one provided by LLM ('answer')\n"

def test_incorrect_outputs_number() -> None:

    prompt = Prompt("Answer to everything").returns("anser", type="int").returns("")

    with pytest.raises(QueryError) as exc_info:
        prompt.query(llm_completion=response_wrong_type, max_retries=1)

    # Check that the error(s) we got are the one(s) we expected
    query_error = exc_info.value
    assert query_error.__str__().startswith(f"Failed LLM query (tried 1 times)")
    assert len(query_error.errors) == 1
    assert query_error.errors[0].__str__() == "Expected 2 outputs in LLM response, but got 1"    
