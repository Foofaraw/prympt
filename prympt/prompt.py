# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

from __future__ import (  # Required for forward references in older Python versions
    annotations,
)

import inspect
import json
import warnings
from typing import Any, List, Tuple, Union

from jinja2 import TemplateSyntaxError

from .exceptions import ConcatenationError, ReplacementError, ResponseError
from .output import Output, outputs_to_xml
from .response import Response
from .utils import extract_jinja_variables, jinja_substitution, litellm_completion

jinja_template_outputs = """

Provide your response inside an XML such as this:

"""


class Prompt:
    """A class representing a prompt template with support for variables and outputs.

    Attributes:
        template (str): The template string containing Jinja variables.
        outputs (List[Output]): List of outputs.
    """

    def __init__(self, template: str, returns: List[Output] = []):
        """Initialize a Prompt instance.

        Args:
            template (str): The template string.
            returns (List[Output]): List of outputs
        """
        self.template: str = template
        self.outputs: List[Output] = returns

    def __call__(self, *args: Any, **kwargs: Any) -> "Prompt":
        """Render the prompt with the given keyword arguments.

        Args:
            *args: Variables to substitute into the template.
            **kwargs: Named variables to substitute into the template.

        Returns:
            Prompt: A new Prompt instance with substituted template.
        """
        variable_names = self.get_variables()

        if len(args) > len(variable_names):
            raise ReplacementError(
                f"Provided {len(args)} positional arguments, but prompt template has only {len(variable_names)} variables"
            )

        for k, v in zip(variable_names, args):

            if k in kwargs:
                raise ReplacementError(
                    f"Got multiple values for template variable '{k}'"
                )
            kwargs[k] = v

        return Prompt(jinja_substitution(self.template, **kwargs), returns=self.outputs)

    def __add__(self, other: Any) -> "Prompt":
        """Concatenate two prompts.

        Args:
            other (Union[str, Prompt]): The prompt or string to concatenate.

        Returns:
            Prompt: A new Prompt instance with combined template and outputs.

        Raises:
            PromptConcatenationError: If trying to add a non-string or non-Prompt object.
        """
        if isinstance(other, str):
            other_prompt = Prompt(other)
        elif isinstance(other, Prompt):
            other_prompt = other
        else:
            raise ConcatenationError(
                "Prompt error: trying to add Prompt to object other than str|Prompt for __add__"
            )

        return Prompt(
            self.template + "\n" + other_prompt.template,
            returns=self.outputs + other_prompt.outputs,
        )

    def __str__(self) -> str:
        """Render the prompt as a string, checking for undefined variables.

        Returns:
            str: The rendered prompt string.
        """

        try:
            variables = extract_jinja_variables(self.template)

            if variables:
                warning = f"Tried to render prompt that still has undefined Jinja2 variables: ({', '.join(sorted(variables))})"
                warnings.warn(warning, RuntimeWarning)
        except TemplateSyntaxError as e:
            warning = f"Tried to render prompt that contains incorrect Jinja2 template syntax ({str(e)})"
            warnings.warn(warning, RuntimeWarning)

        string = self.template

        if self.outputs:
            outputs_copy = self.outputs.copy()
            for output in outputs_copy:
                output.content = "... value for this output goes here ..."

            string += jinja_substitution(
                jinja_template_outputs, outputs=self.outputs
            ) + outputs_to_xml(self.outputs)

        return string

    @classmethod
    def load(cls, template_file: str) -> "Prompt":
        """Load a prompt template from a file.

        Args:
            template_file (str): Path to the template file.

        Returns:
            Prompt: A new Prompt instance with the template content.
        """
        with open(template_file, "r") as file:
            return cls(file.read())

    def get_variables(self) -> List[str]:
        """Extract variables from the template.

        Returns:
            set[Any]: List of variable names present in the template.
        """
        return extract_jinja_variables(self.template)

    def returns(self, *args: Any, **kwargs: Any) -> "Prompt":
        """Add an output to the prompt.

        Args:
            *args: inputs for the Output constructor
            **kwargs: inputs for the Output constructor

        Returns:
            Prompt: A new Prompt instance with the added output.
        """

        return Prompt(self.template, self.outputs + [Output(*args, **kwargs)])

    def query(
        self,
        llm_completion: Any = litellm_completion,
        retries: int = 4,
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        """Query an LLM with the prompt and handle retries.

        Args:
            llm_completion (Callable): The LLM completion function.
            retries (int): Number of retry attempts.
            *args: Additional positional arguments for the LLM function.
            **kwargs: Additional keyword arguments for the LLM function.

        Returns:
            Response: The response from the LLM.

        Raises:
            ResponseError: raised when LLM response and Prompt contain incompatible outputs (different number, name or type).
        """

        response_params = inspect.signature(Response.__init__).parameters

        response_kwargs = {k: v for k, v in kwargs.items() if k in response_params}
        llm_completion_kwargs = {
            k: v for k, v in kwargs.items() if k not in response_params
        }

        for retry_time in range(retries):

            try:

                if retry_time > 0:
                    warnings.warn("Setting temperature to 1!", RuntimeWarning)
                    llm_completion_kwargs["temperature"] = 1.0

                raw_response_text = llm_completion(
                    self.__str__(), *args, **llm_completion_kwargs
                )
                response = Response(raw_response_text, **response_kwargs)

                # Check that expected and responded outputs are compatible
                if len(self.outputs) != len(response):
                    raise ResponseError(
                        f"Expected {len(self.outputs)} outputs in LLM response, but got {len(response)}"
                    )

                errors = ""
                for index, (defined, responded) in enumerate(
                    zip(self.outputs, response)
                ):
                    if defined.name != responded.name:
                        errors += f"Name for output at position {index} ('{defined.name}') differs from the one provided by LLM ('{responded.name}')\n"
                    if defined.type != responded.type:
                        errors += f"Type for output at position {index} ('{defined.type}') differs from the one provided by LLM ('{responded.type}')\n"

                if errors:
                    raise ResponseError(errors)

                break

            except Exception as e:
                warnings.warn(
                    f"WARNING: failed LLM query (try {retry_time} out of {retries}), reason: {str(e)}",
                    RuntimeWarning,
                )
                if retry_time + 1 == retries:
                    raise e

        return response
