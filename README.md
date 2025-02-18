# Prympt: A Python Package for LLM Prompting and Interfacing

Prympt is an open source Python package designed to simplify and standardize interactions with Large Language Models (LLMs). It encapsulates typical boilerplate functionality such as templating, prompt combination, and structured output handling—all in a lightweight package.

Prympt is provided as a free software under MIT license. Feedback and contributions to improve it are welcome!

---

## Overview

Prympt helps to:
- **Compose dynamic prompts:** Use [Jinja2](https://jinja.palletsprojects.com/) syntax to easily substitute variables and iterate over collections.
- **Combine prompts:** Seamlessly merge multiple prompt templates using the `+` operator.
- **Define structured outputs:** Specify expected output formats (e.g., type) so that the responses from LLMs can be automatically verified and parsed.
- **Robust error handling:** Automatically retry and recover from common LLM response errors or malformed outputs.
- **Interface with multiple LLMs:** By default, Prympt integrates with [LiteLLM](https://github.com/BerriAI/litellm), but can switch to other LLM providers easily.

---

## Features

- **Enhanced Jinja2 Templating:** Extends base Jinja2 capabilities with custom functionality to combine multiple templates, ensuring prompts are modular and reusable.
- **Structured Output Definitions:** Annotate prompts with expected outputs. Prympt can automatically verify that the LLM responses match annotated outputs.
- **Type Enforcement:** Define expected types for outputs (e.g., `int`, `float`). Prympt will validate the responses, raise exceptions, and retry queries if the output does not conform.
- **Error Recovery:** Built-in mechanisms to retry LLM queries when provided outputs are not as expected. This makes the tool particularly robust for working with LLMs that might occasionally return malformed data.
- **Flexible LLM Integration:** Whether you use OpenAI, DeepSeek, or another provider, Prympt offers a default interface and the option to specify own LLM completion function.

---

## Installation

Install Prympt from PyPI using pip:

    pip install prympt

### Environment Configuration

Set up your environment by defining the necessary API keys. You can add these to an `.env` file or set them in your environment.

- **For OpenAI:**

      OPENAI_API_KEY=your_openai_api_key_here

- **For DeepSeek:**

      DEEPSEEK_API_KEY=your_deepseek_api_key_here
      LLM_MODEL=deepseek/deepseek-chat

See [LiteLLM providers](https://docs.litellm.ai/docs/providers/) for further info on configuring Prympt with other LLM service providers.

---

## Basic Usage

### Importing and Using the Prompt Class

Prympt’s main entry point is the `Prompt` class. Here’s a simple example that uses it to generate a poem:

    from prympt import Prompt

    model_params = {
        "model": "gpt-4o",
        "temperature": 1.0,
        "max_tokens": 5000,
    }

    response = Prompt("Can you produce a short poem?").query(**model_params)

The response can be printed as a regular string, although it is a Python object of type `Response`:

    print(response)

By default, the `query()` function uses LiteLLM to interact with the chosen LLM. That function does several more things, such as parsing the response of the LLM for return values (see below).

If you prefer to use your own way to interact with the LLM, you can supply a custom completion function to `query()`:

    def custom_llm_completion(prompt: str, *args, **kwargs) -> str:
        # Replace with your own LLM API call
        message = llm(prompt)
        return message

    response = Prompt("Can you produce a short poem?").query(llm_completion=custom_llm_completion, **model_params)
    print(response)

---

## Jinja2 Substitutions

Prympt supports full Jinja2 templating for dynamic prompt generation:

    sms_prompt = Prompt("Hi {{ name }}, your appointment is at {{ time }}.")
    print(sms_prompt(name="Alice", time="2 PM"))

Advance substitutions are also possible (Jinja2 iterations):

    order_prompt = Prompt("""
    Your order includes:
    {% for item in items %}
    - {{ item }}
    {% endfor %}
    """)
    print(order_prompt(items=["Laptop", "Mouse", "Keyboard"]))

---

## Combining Prompts

Prompts can be concatenated using the `+` operator to build more complex interactions.

    greeting = Prompt("Dear {{ customer_name }},\n")
    body = Prompt("We are pleased to inform you that your order (Order #{{ order_number }}) has been shipped and is expected to arrive by {{ delivery_date }}.\n")
    closing = Prompt("Thank you for choosing {{ company_name }}.\nBest regards,\n{{ company_name }} Support Team")

    combined_email_prompt = greeting + body + closing

    print(combined_email_prompt(
        customer_name="Alice Johnson",
        order_number="987654",
        delivery_date="2025-03-25",
        company_name="TechStore"
    ))

---

### Return Value

Prompts can be annotated with expected return values:

    prompt = Prompt("What is the meaning of life, the universe, and everything?")
    response = prompt.returns(name="meaning", type="int").query(**model_params)
    print(response.meaning) # Expected output: 42

Returned values are automatically parsed and attached as member variables to the response. This approach makes it simple to extract and use them.

The call to `query()` will automatically raise errors (or retry, if retries parameter is set to >= 1, see below) when the values provided by the LLM do not match the specified types or number.

---

### Multiple Return Values

Prympt supports prompts with multiple expected return values:

    prompt = Prompt("""
    Summarize the following news article:  {{news_body}} 
    Also, provide a sentiment score (scale from -1 to 1) for the news article.
    """).returns("summary", "A concise summary of the news article").returns(name="sentiment", type="float")

    news_body = "..."
    combined_response = prompt(news_body = news_body).query(**model_params)
    print(combined_response.summary)    # Expected output: A brief summary of the news article
    print(combined_response.sentiment)  # Expected output: A sentiment score between -1 and 1

You can also specify the expected outputs as a list of `Output` objects in the Prompt constructor:

    from prympt import Output

    prompt = Prompt("""
    Summarize the following news article:  {{news_body}} 
    Also, provide a sentiment score (scale from -1 to 1) for the news article.
    """, returns=[
        Output("summary", "A concise summary of the news article"),
        Output(name="sentiment", type="float")
    ])

    news_body = "..."
    response = prompt(news_body = news_body).query(**model_params)
    print(response.summary)    # Expected output: A brief summary of the news article
    print(response.sentiment)  # Expected output: A sentiment score between -1 and 1

---

## Error Control

### Automatic LLM Query Error Recovery

Prympt includes an automatic retry mechanism for queries. You can specify the number of retries if the LLM response does not match the expected output structure:

    prompt = Prompt("Generate Python function that prints weekday, from any given date").returns("python", "python code goes here")
    response = prompt.query(retries=5)  # Default number of retries is 3
    print(response)

### Warnings

Prympt will issue warnings in cases such as:
- Errors during Jinja2 template rendering (e.g., undefined variables or incorrect syntax).
- Transient errors during `Prompt.query()` when retries are in progress.

### Exceptions

Prympt defines a hierarchy of exceptions for granular error handling when retries fail:

- **MalformedOutput:** Raised by `Prompt.returns()` and the `Output` constructor when:
  - The output name is invalid (must be a valid Python identifier: [a-z_][a-z0-9_-]*).
  - The specified type cannot be parsed (must be a valid Python type, e.g., `int`, `float`).
  - The LLM provides a value that cannot be converted to the expected type.
- **ConcatenationError:** Raised when attempting to add a prompt to an unsupported type.
- **ResponseError:** Raised by `Prompt.query()` when the LLM response does not match the expected output structure (e.g., incorrect number, name, or type of outputs).

All these custom exceptions inherit from a common Exception class `PromptError`.

---

## Development

### Setting Up the Development Environment

Install Prympt along with its development dependencies:

    pip install prympt[dev]

### Code Formatting and Linting

Use the following commands to ensure your code adheres to project standards:

    black .
    isort .
    ruff check . --fix
    mypy .

### Running Tests

Execute the test suite with:

    pytest

