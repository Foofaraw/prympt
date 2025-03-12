# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

from __future__ import (  # Required for forward references in older Python versions
    annotations,
)

import copy
import inspect
import warnings
from typing import Any, Dict, List, Callable, Tuple
from dataclasses import dataclass, field

from jinja2 import Environment, StrictUndefined, TemplateSyntaxError, nodes
from jinja2.visitor import NodeVisitor
from litellm import completion

from .exceptions import PrymptError, ConcatenationError, PromptError, ReplacementError, ResponseError
from .output import Output, outputs_to_xml
from .tool_call import any_to_prympt_tool, Tool

_jinja_env = Environment(undefined=StrictUndefined)

def _extract_jinja_variables(template_source: str) -> List[str]:
    class OrderedVariableCollector(NodeVisitor):
        def __init__(self) -> None:
            self.variables: List[str] = []

        def visit_Name(self, node: nodes.Name) -> None:
            # The attribute 'name' may not be recognized by mypy on jinja2.nodes.Name.
            if node.name not in self.variables:
                self.variables.append(node.name)
            # generic_visit may be untyped.
            self.generic_visit(node)

        def visit_For(self, node: nodes.For) -> None:
            # Only visit the 'iter' part. The attribute 'iter' might not be recognized.
            self.visit(node.iter)
            # Skip visiting node.target, node.body, and node.else_ to avoid loop-local variables.

    env = Environment()
    parsed_template = env.parse(template_source)
    collector = OrderedVariableCollector()
    collector.visit(parsed_template)
    return collector.variables


def _jinja_substitution(template: str, **kwargs: Any) -> str:
    """
    Substitutes variables into a Jinja2 template string and returns the rendered result.

    :param template: str - The Jinja2 template string.
    :param kwargs: dict - Variables to substitute into the template.
    :return: str - The rendered template with variables substituted.
    """
    return _jinja_env.from_string(template).render(**kwargs)


def litellm_completion(prompt: str, *args: List[Any], **kwargs: Dict[str, Any]) -> str:
    response = completion(messages=[dict(role="user", content=prompt)], *args, **kwargs)
    return str(response.choices[0].message.content)


class Prompt:
    """A class representing a prompt template with support for variables and outputs.

    Attributes:
        template (str): The template string containing Jinja variables.
        outputs (List[Output]): List of outputs.
    """

    def __init__(self, template: str = "", outputs: List[Output] = [], tools: List[Tool] = []):
        """Initialize a Prompt instance.

        Args:
            template (str): The template string.
            returns (List[Output]): List of outputs
        """
        self.template: str = template
        self.outputs: List[Output] = outputs
        self.tools: Dict[str, Tool] = { tool.name: tool for tool in tools }

        # Make sure there are no outputs with duplicate names
        errors = []
        index_for_name = dict()
        for index, output in enumerate(outputs):
            name = output.name
            if name not in index_for_name:
                index_for_name[name] = index
            else:
                errors += [
                    f"Found outputs at positions {index_for_name[name]}, {index} with same name: '{name}'"
                ]
        if errors:
            raise PromptError("\n".join(errors))
        
    def tool_schemas(self) -> List[Dict[str,Any]]:
        return [ tool.schema for __, tool in self.tools.values() ]

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

        return Prompt(
            _jinja_substitution(self.template, **kwargs),
            outputs=self.outputs,
            tools = self.tools.items(),
        )

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

        overlapping_tools = sorted(self.tools.keys() & other_prompt.tools.keys())
        if overlapping_tools:
            raise ConcatenationError(
                f"Trying to concatenate two prompts with overlapping tools: {', '.join(overlapping_tools)}"
            )
            
        return Prompt(
            self.template + "\n" + other_prompt.template,
            outputs = self.outputs + other_prompt.outputs,
            tools = (self.tools | other_prompt.tools).values(),
        )

    def __str__(self) -> str:
        """Render the prompt as a string, checking for undefined variables.

        Returns:
            str: The rendered prompt string.
        """

        if variables := self.get_variables():
            warning = f"Tried to render prompt that still has undefined Jinja2 variables: ({', '.join(sorted(variables))})"
            warnings.warn(warning, RuntimeWarning)

        string = self.template

        if self.outputs:
            outputs_with_content_indications = copy.deepcopy(self.outputs)
            for output in outputs_with_content_indications:
                if output.name:
                    output.content = (
                        f"... value for output '{output.name}' goes here ..."
                    )
                else:
                    output.content = "... value for this output goes here ..."

            string += (
                "\nProvide your response inside an XML such as this:\n"
                + outputs_to_xml(outputs_with_content_indications)
            )

        return string

    def to_string(self):
        return self.__str__()
       
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

        try:
            return _extract_jinja_variables(self.template)
        except TemplateSyntaxError as e:
            warning = f"Tried to render prompt that contains incorrect Jinja2 template syntax ({str(e)})"
            warnings.warn(warning, RuntimeWarning)
            return []

    def returns(self, *args: Any, **kwargs: Any) -> "Prompt":
        warnings.warn(
            "Prompt.returns() deprecated; use Prompt.output() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.output(*args, **kwargs)

    def output(self, *args: Any, **kwargs: Any) -> "Prompt":
        """Add an output to the prompt.

        Args:
            *args: inputs for the Output constructor
            **kwargs: inputs for the Output constructor

        Returns:
            Prompt: A new Prompt instance with the added output.
        """
        return Prompt(
            self.template,
            self.outputs + [Output(*args, **kwargs)],
            self.tools.items(),
            )

    def tool(self, callable) -> "Prompt":

        tool = Tool(callable)
        
        if tool.name in self.tools:
            raise PromptError(f"Tool '{tool.name}' already exists in prompt")
        
        return Prompt(
            self.template,
            self.outputs,
            list(self.tools.items()) + [tool],
            )

    def error(self, error: PrymptError) -> Prompt:
        return self + f"\n\nMake sure to avoid the following error in your response: {error.message}\n"

    def query(
        self,
        llm_completion: Any = litellm_completion,
        retries: int = 4,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Query an LLM with the prompt and handle retries.

        Args:
            llm_completion (Callable): The LLM completion function.
            retries (int): Number of retry attempts.
            *args: Additional positional arguments for llm_completion.
            **kwargs: Additional keyword arguments for llm_completion.

        Returns:
            Response: The response from the LLM.

        Raises:
            PrymptError: raised when response to LLM call does not conform prompt requirements (e.g. incompatible outputs) for `retries` number of times.
        """

        from .response import Response
        
        prompt, last_error = self, None
        
        for retry_time in range(retries):

            try:
                return Response(llm_completion(prompt.to_string(), *args, **kwargs), prompt)

            except PrymptError as e:
                   
                warn_message = f"WARNING: failed LLM query (try {retry_time} out of {retries}), reason: {e.message} ({e.message})" 
                warnings.warn(warn_message, RuntimeWarning)
                
                prompt = prompt.error(e)
                last_error = e

        raise last_error
    
