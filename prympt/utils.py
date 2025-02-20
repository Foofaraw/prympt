# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

from __future__ import (  # Required for forward references in older Python versions
    annotations,
)

import re
from typing import Any, Dict, List

from jinja2 import Environment, StrictUndefined, nodes
from jinja2.visitor import NodeVisitor
from litellm import completion


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
            if node.name not in self.variables:  # type: ignore[attr-defined]
                self.variables.append(node.name)  # type: ignore[attr-defined]
            # generic_visit may be untyped.
            self.generic_visit(node)  # type: ignore

        def visit_For(self, node: nodes.For) -> None:
            # Only visit the 'iter' part. The attribute 'iter' might not be recognized.
            self.visit(node.iter)  # type: ignore[attr-defined, no-untyped-call]
            # Skip visiting node.target, node.body, and node.else_ to avoid loop-local variables.

    env = Environment()
    parsed_template = env.parse(template_source)
    collector = OrderedVariableCollector()
    collector.visit(parsed_template)  # type: ignore[no-untyped-call]
    return collector.variables


def jinja_substitution(template: str, **kwargs: Any) -> str:
    """
    Substitutes variables into a Jinja2 template string and returns the rendered result.

    :param template: str - The Jinja2 template string.
    :param kwargs: dict - Variables to substitute into the template.
    :return: str - The rendered template with variables substituted.
    """
    return __jinja_env.from_string(template).render(**kwargs)
