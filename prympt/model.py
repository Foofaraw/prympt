import json
from typing import List, Any

from typing import Any, Dict, List
import concurrent.futures

from litellm import (
    completion,
    embedding,
    supports_function_calling,
    supports_parallel_function_calling,
)

from .exceptions import ResponseError, QueryError
from .tool import tools_to_schemas, tools_to_prompt
from .prompt import Prompt

import nest_asyncio
nest_asyncio.apply()

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
    
    def __init__(self, path = None, params = None):
        
        assert path or params
        
        if path:
            with open(path, "r", encoding="utf-8") as f:
                self.params = json.load(f)
        elif params:
            self.params = params

        self.embeddings_cache = dict()
        
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

    def embeddings(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        
        if use_cache:
            not_cached = [ text for text in texts if text not in self.embeddings_cache ]
        else:
            not_cached = texts
        
        response = embedding(
            input=not_cached,
            **self.params
        )
        assert response.model == self.params['model']
        new_embeddings = []
        for idx, entry in enumerate(response.data):
            assert entry['object'] == 'embedding'
            assert entry['index'] == idx
            new_embeddings.append(entry['embedding'])
            
        if not use_cache:
            embeddings = new_embeddings
        else:
            for idx, emb in enumerate(new_embeddings):
                self.embeddings_cache[not_cached[idx]] = emb

            embeddings = []
            for text in texts:
                embeddings.append(self.embeddings_cache[text])

        assert len(texts) == len(embeddings)
        return embeddings


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
                    llm_response = litellm_completion(prompt, tools=tool_schemas, **self.params)
                else:
                    llm_response = litellm_completion(prompt+tool_calling_prompt, **self.params)

                response = Response(llm_response, prompt, tools = tools)
                response.errors = errors
                
                return response

            except ResponseError as e:
                prompt = prompt.error(e)
                errors.append(e)

        raise QueryError(prompt+tool_calling_prompt, errors)
    