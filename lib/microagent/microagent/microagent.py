"""
Self-contained MicroAgent base class.

Unifies the agent interface, configuration, LLM interaction, and execution loop
into a single class. Subclasses implement problem-specific logic; the base class
handles everything else.
"""

import asyncio
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Tuple, Callable, List, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import litellm
except ImportError:
    litellm = None  # Allow import without litellm for subclasses that override _call_llm

try:
    import yaml
except ImportError:
    yaml = None  # Allow import without yaml for dict-based config


class MicroAgent(ABC):
    """
    Abstract base class for self-contained micro agents.

    Combines the agent interface, configuration, LLM interaction, and execution
    loop into a single class. Subclasses only need to implement problem-specific
    logic (prompts, parsing, state updates, termination).

    Configuration can be provided via:
    - config_file: Path to a YAML configuration file
    - config_dict: A dictionary with configuration
    - **kwargs: Override individual parameters

    Example usage:
        class MyAgent(MicroAgent):
            def create_initial_state(self): return {}
            def generate_step_prompt(self, state): return "..."
            def update_state(self, state, result): return {**state, 'result': result}
            def is_solved(self, state): return state.get('done', False)
            def get_problem_complexity(self, state): return 1

        agent = MyAgent(config_file="config.yaml")
        trace = await agent.execute()
    """

    # Default system prompt - subclasses can override
    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

    def __init__(
        self,
        config_file: Optional[str] = None,
        config_dict: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize the agent with configuration.

        Args:
            config_file: Path to YAML config file
            config_dict: Configuration dictionary
            **kwargs: Override specific config values (model, temperature, etc.)
        """
        self._load_config(config_file, config_dict, **kwargs)
        self.total_cost = 0.0
        self.total_api_calls = 0
        logger.info(f"Initialized {self.__class__.__name__} with model: {self.model}")

    def _load_config(
        self,
        config_file: Optional[str],
        config_dict: Optional[Dict],
        **kwargs
    ):
        """Load and process configuration from file, dict, or kwargs."""
        config = {}

        # Load from file if provided
        if config_file is not None:
            if yaml is None:
                raise ImportError("pyyaml is required for config file loading. Install with: pip install pyyaml")
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        elif config_dict is not None:
            config = config_dict

        # Extract nested config sections
        model_config = config.get('model', {})
        defaults = config.get('litellm_defaults', {})

        # Build model name from provider/name if needed
        provider = model_config.get('provider')
        model_name = model_config.get('name')
        if provider and model_name and '/' not in model_name:
            full_model_name = f"{provider}/{model_name}"
        else:
            full_model_name = model_name

        # Helper for hierarchical config lookup: kwargs > model_config > defaults > fallback
        def get_value(key: str, fallback: Any) -> Any:
            return kwargs.get(key, model_config.get(key, defaults.get(key, fallback)))

        # Core model settings
        self.model = kwargs.get('model', full_model_name)
        if self.model is None:
            raise ValueError("Model name is required. Provide via config file, config_dict, or 'model' kwarg.")

        # LLM parameters
        self.temperature = get_value('temperature', 0.7)
        self.max_tokens = get_value('max_tokens', 2048)
        self.top_p = get_value('top_p', 1.0)
        self.frequency_penalty = get_value('frequency_penalty', 0.0)
        self.presence_penalty = get_value('presence_penalty', 0.0)

        # Cost tracking
        self.cost_per_input_token = model_config.get('cost_per_input_token', 0.0)
        self.cost_per_output_token = model_config.get('cost_per_output_token', 0.0)

        # Retry configuration
        self.retry_delays = model_config.get('retry_delays', [1, 2, 4, 8])

        # Store raw config for subclass access
        self.config = config

    # ─────────────────────────────────────────────────────────────────────────
    # Abstract Methods (subclasses must implement)
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def create_initial_state(self, *args, **kwargs) -> Any:
        """Create the initial state for the problem."""
        pass

    @abstractmethod
    def generate_step_prompt(self, state: Any) -> str:
        """Generate the user prompt for the next step based on current state."""
        pass

    @abstractmethod
    def update_state(self, current_state: Any, step_result: Any) -> Any:
        """Update the state based on the step result."""
        pass

    @abstractmethod
    def is_solved(self, state: Any) -> bool:
        """Check if the problem is solved."""
        pass

    @abstractmethod
    def get_problem_complexity(self, state: Any) -> int:
        """
        Get the problem complexity for calibration purposes.

        Returns:
            int: A positive integer representing problem complexity
        """
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Overridable Hooks
    # ─────────────────────────────────────────────────────────────────────────

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for LLM calls.

        Override this method to customize the system prompt for your agent.
        Default returns DEFAULT_SYSTEM_PROMPT.
        """
        return self.DEFAULT_SYSTEM_PROMPT

    def get_response_parser(self) -> Callable[[str], Any]:
        """
        Get the response parser for this agent.

        Override this to provide domain-specific parsing of LLM responses.
        Default returns a passthrough (identity) function.
        """
        return lambda x: x

    def validate_step_result(self, step_result: Any) -> bool:
        """
        Validate that a step result is acceptable before updating state.

        Override for domain-specific validation.
        Default checks that result is not None.
        """
        return step_result is not None

    def step_generator(self, state: Any) -> Tuple[Tuple[str, str], Callable[[str], Any]]:
        """
        Generate the prompt tuple and parser for the current state.

        Returns:
            Tuple of ((system_prompt, user_prompt), response_parser)

        Override this for full control over prompt generation.
        Default implementation calls get_system_prompt(), generate_step_prompt(),
        and get_response_parser().
        """
        system_prompt = self.get_system_prompt()
        user_prompt = self.generate_step_prompt(state)
        parser = self.get_response_parser()
        return (system_prompt, user_prompt), parser

    # ─────────────────────────────────────────────────────────────────────────
    # Execution (built-in)
    # ─────────────────────────────────────────────────────────────────────────

    async def execute(self, *args, **kwargs) -> List[Any]:
        """
        Execute the agent to solve the problem.

        Args:
            *args, **kwargs: Passed to create_initial_state()

        Returns:
            List of states representing the execution trace
        """
        logger.info(f"Starting execution with args={args}, kwargs={kwargs}")

        state = self.create_initial_state(*args, **kwargs)
        trace = [state]

        while not self.is_solved(state):
            prompt_tuple, parser = self.step_generator(state)
            raw_result = await self._call_llm(prompt_tuple)
            parsed_result = parser(raw_result)

            if not self.validate_step_result(parsed_result):
                logger.warning(f"Step result validation failed: {parsed_result}")

            state = self.update_state(state, parsed_result)
            trace.append(state)
            logger.info("State updated.")

        logger.info(f"Execution completed. Trace length: {len(trace)} states")
        return trace

    async def _call_llm(self, prompt_tuple: Tuple[str, str]) -> str:
        """
        Call the LLM with retry logic.

        Args:
            prompt_tuple: (system_prompt, user_prompt)

        Returns:
            Raw string response from the LLM

        Override this method to use a different LLM backend or custom logic.
        """
        if litellm is None:
            raise ImportError("litellm is required for LLM calls. Install with: pip install litellm")

        system_prompt, user_prompt = prompt_tuple
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        llm_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

        last_exception = None
        for attempt, delay in enumerate(self.retry_delays):
            try:
                self.total_api_calls += 1
                logger.info(f"Calling LLM (Attempt {attempt + 1}/{len(self.retry_delays)})...")

                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    **llm_kwargs
                )

                if response is None or response.choices is None or len(response.choices) == 0:
                    raise ValueError("Received an empty or invalid response from the LLM.")

                raw_result = response.choices[0].message.content
                if raw_result is None:
                    raise ValueError("The LLM response content was empty.")

                logger.info(f"LLM response received: '{raw_result[:100]}...'")

                # Track cost if available
                if hasattr(response, '_hidden_params') and 'response_cost' in response._hidden_params:
                    self.total_cost += response._hidden_params['response_cost']

                return raw_result

            except Exception as e:
                last_exception = e
                logger.warning(f"LLM call failed on attempt {attempt + 1}: {e}")
                if attempt < len(self.retry_delays) - 1:
                    # Add jitter to avoid retry storms with parallel agents
                    jittered_delay = delay + random.random()
                    logger.info(f"Retrying in {jittered_delay:.2f} seconds...")
                    await asyncio.sleep(jittered_delay)

        logger.error("All retry attempts failed.")
        raise last_exception or RuntimeError("LLM call failed after all retries")
