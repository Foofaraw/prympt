# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

from __future__ import (  # Required for forward references in older Python versions
    annotations,
)

import re
from typing import Any, Dict, List

from jinja2 import Environment, StrictUndefined, meta
from litellm import completion


def litellm_completion(prompt: str, *args: List[Any], **kwargs: Dict[str, Any]) -> str:
    response = completion(messages=[dict(role="user", content=prompt)], *args, **kwargs)
    return str(response.choices[0].message.content)


__jinja_env = Environment(undefined=StrictUndefined)


def extract_jinja_variables(template_str: str) -> set[Any]:
    """
    Extracts undeclared variables from a Jinja2 template string.

    :param template_str: str - The Jinja2 template string to analyze.
    :return: set - A set of variable names found in the template.
    """
    ast = __jinja_env.parse(
        template_str
    )  # Parse the template into an abstract syntax tree (AST)
    return meta.find_undeclared_variables(ast)  # Extract variables


def jinja_substitution(template: str, **kwargs: Any) -> str:
    """
    Substitutes variables into a Jinja2 template string and returns the rendered result.

    :param template: str - The Jinja2 template string.
    :param kwargs: dict - Variables to substitute into the template.
    :return: str - The rendered template with variables substituted.
    """
    return __jinja_env.from_string(template).render(**kwargs)
