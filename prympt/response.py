# Copyright (c) 2025 foofaraw (GitHub: foofaraw)
# Licensed under the MIT License (see LICENSE file for details).

from __future__ import (  # Required for forward references in older Python versions
    annotations,
)

from typing import Iterator, List

from .output import Output, xml_to_outputs


class Response:
    """
    A class representing a response containing code blocks.

    Attributes:
        __raw_response_text (str): The raw response text.
        __outputs (List[Output]): The extracted code blocks.
    """

    def __init__(self, response_text: str):
        """
        Initializes a Response object.

        Args:
            response_text (str): The raw response text.
        """
        self.__raw_response_text: str = response_text

        self.__outputs: List[Output] = xml_to_outputs(self.__raw_response_text)

        # Add output contents as member variables in response object
        for output in self.__outputs:
            if output.name and not hasattr(self, output.name):
                setattr(self, output.name, output.content)

    def __str__(self) -> str:
        """Returns the raw response text."""
        return self.__raw_response_text

    def __iter__(self) -> Iterator[Output]:
        """Returns an iterator over the code blocks."""
        return iter(self.__outputs)

    def __len__(self) -> int:
        """Returns the number of code blocks."""
        return len(self.__outputs)

    def __getitem__(self, index: int) -> Output:
        """Returns the code block at the given index."""
        return self.__outputs[index]

    def __contains__(self, name: str) -> bool:
        """
        Checks if a code block with the given language exists.

        Args:
            name (str): The output name.

        Returns:
            bool: True if output with that name exists in response, False otherwise.
        """
        return any(output.name == name for output in self.__outputs)
