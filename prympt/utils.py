# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

from __future__ import (  # Required for forward references in older Python versions
    annotations,
)

import ast
from typing import Any, Dict, List

from jinja2 import Environment, StrictUndefined, nodes
from jinja2.visitor import NodeVisitor
from litellm import completion

from .exceptions import MalformedOutput, ResponseError


def litellm_completion(prompt: str, *args: List[Any], **kwargs: Dict[str, Any]) -> str:
    response = completion(messages=[dict(role="user", content=prompt)], *args, **kwargs)
    return str(response.choices[0].message.content)


__jinja_env = Environment(undefined=StrictUndefined)


def extract_jinja_variables(template_source: str) -> List[str]:
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


def jinja_substitution(template: str, **kwargs: Any) -> str:
    """
    Substitutes variables into a Jinja2 template string and returns the rendered result.

    :param template: str - The Jinja2 template string.
    :param kwargs: dict - Variables to substitute into the template.
    :return: str - The rendered template with variables substituted.
    """
    return __jinja_env.from_string(template).render(**kwargs)


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
