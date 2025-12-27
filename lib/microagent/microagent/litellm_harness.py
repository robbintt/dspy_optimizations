import asyncio
import logging
import os
from typing import Any, List, Callable, Tuple, Dict, Optional
from .protocols import ExecutionHarness
from . import MicroAgent

try:
    import litellm
except ImportError:
    raise ImportError("The 'litellm' package is required. Please install it with 'pip install litellm'.")

try:
    import yaml
except ImportError:
    raise ImportError("The 'pyyaml' package is required. Please install it with 'pip install pyyaml'.")

class LiteLLMConfig:
    """
    Configuration for the LiteLLMHarness, loaded from a YAML file or config dict.
    """
    def __init__(self, config_file: Optional[str] = None, config_dict: Optional[Dict] = None, **kwargs):
        """
        Initializes configuration from a specified YAML file or config dict.
        
        Args:
            config_file: The absolute or relative path to the YAML config file.
            config_dict: A dictionary containing the configuration.
        """
        if config_file is not None and config_dict is not None:
            raise ValueError("Cannot provide both config_file and config_dict.")
        if config_file is None and config_dict is None:
            raise ValueError("Must provide either a config_file or a config_dict.")
            
        if config_file is not None:
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = config_dict

        self._process_config(config, **kwargs)

    def _process_config(self, config: Dict, **kwargs):
        model_config = config.get('model', {})
        defaults = config.get('litellm_defaults', {})

        provider = model_config.get('provider')
        model_name = model_config.get('name')

        # Construct the full model name for LiteLLM (e.g., 'cerebras/zai-glm-4.6')
        if provider and model_name and '/' not in model_name:
            full_model_name = f"{provider}/{model_name}"
        else:
            # Fallback to the name in config if provider prefix seems unnecessary/already present
            full_model_name = model_name

        self.model = kwargs.get('model', full_model_name)
        if self.model is None:
            raise ValueError("Model name could not be determined.")
            
        self.temperature = kwargs.get('temperature', model_config.get('temperature', defaults.get('temperature', 0.7)))
        self.max_tokens = kwargs.get('max_tokens', model_config.get('max_tokens', defaults.get('max_tokens', 2048)))
        self.top_p = kwargs.get('top_p', model_config.get('top_p', defaults.get('top_p', 1.0)))
        self.frequency_penalty = kwargs.get('frequency_penalty', model_config.get('frequency_penalty', defaults.get('frequency_penalty', 0.0)))
        self.presence_penalty = kwargs.get('presence_penalty', model_config.get('presence_penalty', defaults.get('presence_penalty', 0.0)))

        # Cost tracking (per million tokens)
        self.cost_per_input_token = model_config.get('cost_per_input_token', 0.00015)
        self.cost_per_output_token = model_config.get('cost_per_output_token', 0.0006)

logger = logging.getLogger(__name__)

class LiteLLMHarness(ExecutionHarness):
    """
    An execution harness that uses the 'litellm' library to interact
    with various LLM providers.
    """

    def __init__(self, config: LiteLLMConfig):
        """
        Initialize the harness with a configuration object.

        Args:
            config: A LiteLLMConfig instance with model and parameter settings.
        """
        if not isinstance(config, LiteLLMConfig):
            raise TypeError("config must be an instance of LiteLLMConfig")
            
        self.config = config
        self.model = config.model
        self.llm_kwargs = {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
        }
        self.total_cost = 0.0
        self.total_api_calls = 0
        logger.info(f"Initialized {self.__class__.__name__} with model: {self.model}")

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
