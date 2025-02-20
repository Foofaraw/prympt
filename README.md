# prympt: A Python Package for LLM Prompting and Interfacing

`prympt` is an open source Python package designed to simplify and standardize typical interactions with Large Language Models (LLMs). It encapsulates typical boilerplate functionality for prompt composition and LLM response parsing, such as templating, prompt combination, and structured output handling-all in a lightweight package.

This package is provided as a free software under MIT license. Feedback and contributions to improve it are welcome!

---

## Quick Overview

This is an example that showcases the main features of `prympt`. The following code composes several prompts, defines structured outputs for them, and combines them into a single prompt:

    from prympt import Prompt

    # Define a prompt with annotated output named 'title'
    prompt_title = Prompt(
        "Provide a title for the following movie review: {{movie_review}}",
    ).returns("title")

    # Define a prompt with annotated 'float' output named 'sentiment'
    prompt_sentiment = Prompt(
        "Provide a sentiment score (scale from -1 to 1)."
    ).returns("sentiment", type="float")

    prompt = prompt_title + prompt_sentiment

You can give the template parameter 'movie_review' a specific value, and query the LLM with the prompt. The outputs are easily retrieved as properties of the response object:

    # Define value for 'movie_review'
    movie_review = "A captivating drama that deftly blends mystery with heartfelt emotion. The film follows the story of a troubled detective, Alex Monroe, as he unravels a decades-old mystery that forces him to confront his own past. The narrative is rich with twists and turns, keeping the audience engaged from start to finish."

    # Set the template variable with that value, and query the LLM with the resulting prompt
    response = prompt(movie_review).query(**model_params)
    
    print(response.title)      # Expected output: A one-line title for the review.
    print(response.sentiment)  # Expected output: A sentiment score between -1 and 1.

To summarize, the package provides these main functionalities:

- **Dynamic Prompt Composition:** Leverage enhanced [Jinja2](https://jinja.palletsprojects.com/) templating to easily substitute variables, iterate over collections.
- **Structured Output Definitions:** Annotate your prompts with expected output formats with types (e.g., `int`, `float`) so that responses from LLMs can be automatically verified, parsed, and validated.
- **Combine prompts:** Seamlessly combine multiple prompt templates and their outputs using the `+` operator for modular, reusable prompts.
- **Robust Error Handling:** Built-in mechanisms automatically retry and recover from common LLM response errors or malformed outputs, ensuring reliable interactions even when outputs deviate from expectations.
- **Flexible LLM Integration:** `prympt` integrates by default with [LiteLLM](https://github.com/BerriAI/litellm), which supports over 100 LLM APIs, and also allows you to connect to any LLM API using custom code or your preferred provider.

---

## Installation

Install from PyPI using pip:

    pip install prympt

### Environment Configuration

Set up your environment by defining the necessary API keys. You can add these to an `.env` file or set them in your environment.

- **For OpenAI:**

      OPENAI_API_KEY=your_openai_api_key_here

- **For DeepSeek:**

      DEEPSEEK_API_KEY=your_deepseek_api_key_here
      LLM_MODEL=deepseek/deepseek-chat

See [LiteLLM providers](https://docs.litellm.ai/docs/providers/) for further info on configuring `prympt` with other LLM service providers.

---

## Composing Prompts

### Creating a Prompt Object

`prympt`’s main entry point is the `Prompt` class. Here’s a simple example that uses it to compose a prompt that creates a poem:

    from prympt import Prompt

    prompt = Prompt("Can you produce a short poem?")

### Prompts With Jinja2 Variables

`prympt` supports full Jinja2 templating for dynamic prompt generation:

    sms_prompt = Prompt("Hi {{ name }}, your appointment is at {{ time }}.")

    print(sms_prompt(name="Alice", time="2 PM"))

Advanced substitutions are also possible (Jinja2 iterations):

    order_prompt = Prompt("""
    Your order includes:
    {% for item in items %}
    - {{ item }}
    {% endfor %}
    """)

    print(order_prompt(items=["Laptop", "Mouse", "Keyboard"]))

### Combining Prompts

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

### Annotating Prompts with Outputs

Prompts can be annotated with expected outputs using the `returns` method:

    prompt = Prompt("What is the meaning of life, the universe, and everything?")
    prompt = prompt.returns(name="meaning", type="int").query(**model_params)

The method `returns` has the same parameters as the object `Output`constructor:

- name (str): name of the output (and name of property in the `Response`object)
- description (str): description of output
- type (str): expected type of the output (e.g. `int`, `float`, `str`)

Each prompt can have multiple output annotations:

    prompt = Prompt("""
    Summarize the following news article:  {{news_body}} 
    Also, provide a sentiment score (scale from -1 to 1) for the news article.
    """).returns("summary", "A concise summary of the news article").returns(name="sentiment", type="float")

Outputs can also be specified as a list of `Output` objects in the Prompt constructor:

    from prympt import Output

    prompt = Prompt("""
    Summarize the following news article:  {{news_body}} 
    Also, provide a sentiment score (scale from -1 to 1) for the news article.
    """, returns=[
        Output("summary", "A concise summary of the news article"),
        Output(name="sentiment", type="float")
    ])

---

## Querying Prompts

We can extend the previous example to query the LLM with the prompt:

    # Define LLM model params
    model_params = {
        "model": "gpt-4o",
        "temperature": 1.0,
        "max_tokens": 5000,
    }

    # Define the value for the template variable `news_body`
    news_body = "Aliens attack Earth right after world peace achieved"

    # Query LLM with the prompt
    response = prompt(news_body = news_body).query(**model_params)
    print(response.summary)    # Expected output: A brief summary of the news article
    print(response.sentiment)  # Expected output: A sentiment score between -1 and 1

    response = Prompt("Can you produce a short poem?").query(**model_params)

The method `query` does several more things, such as parsing the response of the LLM for return values (see below). It returns a `Response` object that contains the prompt outputs as member variables. This approach makes it simple to extract and use them.

### Automatic Query Recovery

`prympt` includes an automatic retry mechanism for queries. You can specify the number of retries if the LLM response does not match the expected output structure:

    prompt = Prompt("Generate Python function that prints weekday, from any given date").returns("python", "python code goes here")
    response = prompt.query(retries=5, **model_params)  # Default number of retries is 3
    print(response)

When the `retries` parameter of the `query` method is set to >= 1, the call to `query` will automatically retry the call to the LLM if the LLM's does not reply with the correct outputs (e.g. the LLM provided outputs that cannot be parsed, or do not match the prompt's outputs).

When the `query` method runs out of retries, it will raise an **ResponseError** exception, indicating the last error found in the LLM's response (see below).

### Custom LLM Interfacing

By default `query` uses LiteLLM to interact with the chosen LLM.

If you prefer to use your own way to interact with the LLM, you can supply a custom completion function to `query`:

    def custom_llm_completion(prompt: str, *args, **kwargs) -> str:
        # Replace with your own LLM API call
        message = llm(prompt)
        return message

    response = Prompt("Can you produce a short poem?").query(llm_completion=custom_llm_completion, **model_params)

    print(response)

---

## Error Control

### Warnings

`prympt` will issue warnings in cases such as:
- Errors during Jinja2 template rendering (e.g., undefined variables or incorrect syntax).
- Transient errors during `Prompt.query` when retries are in progress.

### Exceptions

`prympt` defines a hierarchy of exceptions for granular error handling when retries fail:

- **MalformedOutput:** Raised by `Prompt.returns` and the `Output` constructor when:
  - The output name is invalid (must be a valid Python identifier: [a-z_][a-z0-9_-]*).
  - The specified type cannot be parsed (must be a valid Python type, e.g., `int`, `float`).
  - The LLM provides a value that cannot be converted to the expected type.
- **ConcatenationError:** Raised when attempting to add a prompt to an unsupported type.
- **ResponseError:** Raised by `Prompt.query` when the LLM response does not match the expected output structure (e.g., incorrect number, name, or type of outputs).

All these custom exceptions inherit from a common Exception class `PromptError`.

---

## Development

### Setting Up the Development Environment

Install `prympt` along with its development dependencies:

    pip install prympt[dev]

### Code Formatting and Linting

Use the following commands to ensure your code adheres to project standards:

    black .
    isort .
    ruff check . --fix
    mypy .

### Running Tests

Execute the test suite with:

    pytest .
