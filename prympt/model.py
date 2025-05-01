import json
from typing import List, Any

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


class Model:
    
    def __init__(self, model_file = None, model_params = None):
        
        assert model_file or model_params
        
        if model_file:
            with open(model_file, "r", encoding="utf-8") as f:
                self.model_params = json.load(f)
        elif model_params:
            self.model_params = model_params
        
    def __call__(
        self,
        prompt:Prompt,
        max_retries: int = 4,
        tools: List[Any] = [],
        ):
        '''
        Generates response to a prompt
        '''
        return prompt.query(max_retries = max_retries, tools = tools, **self.model_params)

        
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
