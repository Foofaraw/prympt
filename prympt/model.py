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

    def embeddings(self, texts = List[str]) -> List[List[float]]:
        
        import nest_asyncio
        
        # Function to run the embedding in a separate thread with the patched event loop
        def run_embedding_in_thread():
            # Patch the event loop inside this thread
            nest_asyncio.apply()

            # Define your synchronous function for embedding (no need for 'await')
            response = embedding(
                input=texts,
                **self.params
            )
            
            # Sanity checks and embeddings retrieval
            assert response.model == self.params['model']
            embeddings = []
            for idx, entry in enumerate(response.data):
                assert entry['object'] == 'embedding'
                assert entry['index'] == idx
                embeddings.append(entry['embedding'])
            
            assert len(texts) == len(embeddings)
            return embeddings

        # Run the code in a separate thread and get the result
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_embedding_in_thread)
            embedding_result = future.result()  # Get the result from the thread
        
        return embedding_result  # Return the embedding result


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
    