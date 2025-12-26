import asyncio
import logging
from typing import Any, List, Callable, Tuple, Dict
from .protocols import ExecutionHarness
from . import MicroAgent

try:
    import litellm
except ImportError:
    raise ImportError("The 'litellm' package is required. Please install it with 'pip install litellm'.")

logger = logging.getLogger(__name__)

class LiteLLMHarness(ExecutionHarness):
    """
    An execution harness that uses the 'litellm' library to interact
    with various LLM providers.
    """

    def __init__(self, model: str, **llm_kwargs):
        """
        Initialize the harness with a model name and options.

        Args:
            model: The model identifier for litellm (e.g., "gpt-3.5-turbo", "anthropic/claude-3").
            **llm_kwargs: Additional keyword arguments to pass to litellm.completion().
        """
        self.model = model
        self.llm_kwargs = llm_kwargs
        self.total_cost = 0.0
        self.total_api_calls = 0
        logger.info(f"Initialized {self.__class__.__name__} with model: {model}")

    async def execute_step(self,
                          step_prompt: Tuple[str, str],
                          response_parser: Callable[[str], Any]) -> Any:
        """
        Execute a single step by calling the LLM via litellm.

        Args:
            step_prompt: A tuple of (system_prompt, user_prompt).
            response_parser: A function to parse the LLM's response.

        Returns:
            The parsed result from the LLM.
        """
        self.total_api_calls += 1
        system_prompt, user_prompt = step_prompt
        
        logger.info(f"Calling LLM...")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                **self.llm_kwargs
            )
            
            raw_result = response.choices[0].message.content
            logger.info(f"LLM response received: '{raw_result[:100]}...'")
            
            # Update cost tracking
            if hasattr(response, '_hidden_params') and 'response_cost' in response._hidden_params:
                self.total_cost += response._hidden_params['response_cost']
            
            return response_parser(raw_result)
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def execute_plan(self,
                          initial_state: Any,
                          step_generator: Callable[[Any], Tuple[Tuple[str, str], Callable]],
                          termination_check: Callable[[Any], bool],
                          agent: MicroAgent) -> List[Any]:
        """
        Execute a complete plan by repeatedly calling the LLM.
        """
        trace = [initial_state]
        current_state = initial_state

        while not termination_check(current_state):
            prompt, parser = step_generator(current_state)
            result = await self.execute_step(prompt, parser)
            
            current_state = agent.update_state(current_state, result)
            trace.append(current_state)
            logger.info(f"State updated.")

        return trace
