import json
from typing import List, Any

from typing import Any, Dict, List
from litellm import completion, supports_function_calling, supports_parallel_function_calling

from .exceptions import ResponseError, QueryError
from .tool import tools_to_schemas, tools_to_prompt
from .prompt import Prompt

prompt_query = Prompt("""

Given this text:

```
{{text}}
```

Retrieve the following information:

```
{{target}}
```

""")

prompt_update = Prompt("""

Update this text:

```
{{text}}
```

Following these instructions:

```
{{instructions}}
```

""")

def litellm_completion(
    data: Any, # Either string, message or prompt
    *args: List[Any],
    **kwargs: Dict[str, Any]
    ) -> str:
    
    if isinstance(data, str):
        message = dict(role="user", content=data)
    elif isinstance(data, Prompt):
        message = dict(role="user", content=data.__str__())
    else:
        assert isinstance(message, dict)
        message = dataclass

    response = completion(messages=[message], num_retries=0, *args, **kwargs)

    return response.choices[0].message

class Model:
    
    def __init__(self, model_file = None, model_params = None):
        
        assert model_file or model_params
        
        if model_file:
            with open(model_file, "r", encoding="utf-8") as f:
                self.model_params = json.load(f)
        elif model_params:
            self.model_params = model_params

        
    def query(self, text:str, question:str) -> str:
        '''
        Provides the answer to a question specific to the content of a text
        '''
        prompt = prompt_query(
            text = text,
            target = question,
            )
        
        prompt = prompt.returns("result", f"Provide here the following info: {question}")
        result = self(prompt)
        
        return result.result

    def update(self, text:str, instructions:str) -> str:
        '''
        Modifies a text following some given instructions
        '''
        prompt = prompt_update(
            text = text,
            instructions = instructions,
            )
        
        prompt = prompt.returns("updated_text", "Text updated according to the instructions")
        prompt = prompt.returns("changes_summary", "Summary of the changes applied to the text")
        result = self(prompt)
        
        return result.updated_text, result.changes_summary

    def __call__(
        self,
        prompt: Prompt,
        max_retries: int = 4,
        tools: List[Any] = [],
    ) -> Any:
        """Query an LLM with the prompt and handle retries.

        Args:
            prompt (Prompt): The prompt to use in the call.
            retries (int): Number of retry attempts.
            tools (List): List of tools available.
            
        Returns:
            Response: The response from the LLM.

        Raises:
            PrymptError: raised when response to LLM call does not conform prompt requirements (e.g. incompatible outputs) for `retries` number of times.
        """

        from .response import Response
        
        tool_schemas = tools_to_schemas(tools)
        tool_calling_prompt = tools_to_prompt(tools)
        
        errors = []
        
        for __ in range(max_retries):
            
            try:
                
                native_tool_calling = False
                '''
                native_tool_calling = (
                        supports_function_calling(model=kwargs['model']) and
                        supports_parallel_function_calling(model=kwargs['model'])
                    ) if 'model' in kwargs else False
                '''
                if native_tool_calling:
                    llm_response = litellm_completion(prompt, tools=tool_schemas, **self.model_params)
                else:
                    llm_response = litellm_completion(prompt+tool_calling_prompt, **self.model_params)

                response = Response(llm_response, prompt, tools = tools)
                response.errors = errors
                
                return response

            except ResponseError as e:
                prompt = prompt.error(e)
                errors.append(e)

        raise QueryError(prompt+tool_calling_prompt, errors)
    