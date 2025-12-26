"""
Generic executor for running MicroAgent implementations with the MDAP framework.
Provides a clean, reusable interface for executing any MicroAgent-based solver.
"""

import logging
from typing import Any, List, Optional, Callable
import os
import yaml
import litellm
from pathlib import Path
from micro_agent import MicroAgent
from mdap_harness import MDAPHarness

litellm.set_verbose = os.getenv("LITELLM_LOG", "INFO").upper() == "DEBUG"
litellm.drop_params = True

class LLMClient:
    """A simple client wrapper for litellm."""
    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 2048):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def agenerate(self, prompt: str) -> str:
        """Generate a single response from the LLM."""
        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

class MicroAgentConfig:
    """Configuration loaded from microagent's own config file."""
    def __init__(self, **kwargs):
        config_file = Path(__file__).parent / "config" / "models.yaml"
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        model_config = config['model']
        defaults = config['mdap_defaults']
        
        # Allow override via kwargs
        self.model = kwargs.get('model', f"{model_config['provider']}/{model_config['name']}")
        self.temperature = kwargs.get('temperature', model_config.get('temperature', 0.7))
        self.max_tokens = kwargs.get('max_tokens', model_config.get('max_tokens', 2048))
        self.k_margin = kwargs.get('k_margin', defaults['k_margin'])
        self.max_candidates = kwargs.get('max_candidates', defaults['max_candidates'])
        self.max_retries = defaults['max_retries']
        
        # Cost tracking (per million tokens)
        self.cost_per_input_token = model_config.get('cost_per_input_token', 0.00015)
        self.cost_per_output_token = model_config.get('cost_per_output_token', 0.0006)

logger = logging.getLogger(__name__)


class MicroAgentExecutor:
    """
    Generic executor for MicroAgent implementations.

    Provides a unified interface for executing any MicroAgent with the MDAP framework,
    handling configuration, harness setup, and execution orchestration.

    Example usage:
        # Create a specific agent implementation
        agent = HanoiMDAP(config)

        # Create executor and run
        executor = MicroAgentExecutor(agent)
        result = await executor.execute(num_disks=5)

        # Access execution statistics
        print(f"Total cost: ${executor.total_cost:.4f}")
        print(f"API calls: {executor.total_api_calls}")
    """

    def __init__(self, agent: MicroAgent, config: Optional[MicroAgentConfig] = None):
        """
        Initialize the executor with a MicroAgent implementation.

        Args:
            agent: The MicroAgent instance to execute
            config: Optional MicroAgentConfig. If None, creates default
        """
        self.agent = agent

        # Use provided config, or create default
        if config is not None:
            self.config = config
        else:
            self.config = MicroAgentConfig()

        # Create harness for execution
        llm_client = LLMClient(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        self.harness = MDAPHarness(config=self.config, llm_client=llm_client)

        logger.info(f"Initialized MicroAgentExecutor with agent: {agent.__class__.__name__}")
        logger.info(f"Configuration: model={self.config.model}, k_margin={self.config.k_margin}, "
                   f"max_candidates={self.config.max_candidates}")

    async def execute(self, *args, **kwargs) -> List[Any]:
        """
        Execute the microagent with the given arguments.

        Args:
            *args, **kwargs: Arguments passed to agent.create_initial_state()

        Returns:
            List of states representing the execution trace

        Example:
            # For Hanoi solver
            trace = await executor.execute(num_disks=5)

            # For other agents with different initialization
            trace = await executor.execute(problem_data=data, config=params)
        """
        logger.info(f"Starting execution with args={args}, kwargs={kwargs}")

        try:
            # Execute the agent using the harness
            trace = await self.harness.execute_agent_mdap(self.agent, *args, **kwargs)

            logger.info(f"Execution completed successfully")
            logger.info(f"Trace length: {len(trace)} states")
            logger.info(f"Total cost: ${self.total_cost:.4f}")
            logger.info(f"Total API calls: {self.total_api_calls}")

            return trace

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            raise

    async def calibrate(self, calibration_states: List[Any]) -> dict:
        """
        Calibrate the agent using pre-generated states to estimate optimal k_margin.

        Args:
            calibration_states: List of states to use for calibration

        Returns:
            Dictionary with calibration results:
                - 'p_estimate': Estimated per-step success rate
                - 'k_min': Recommended k_margin value
                - 'successful_steps': Number of successful steps
                - 'total_steps': Total steps attempted
        """
        logger.info(f"Starting calibration with {len(calibration_states)} states")

        # Estimate per-step success rate
        p_estimate = await self.harness.estimate_per_step_success_rate_from_states(
            self.agent, calibration_states
        )

        # Calculate recommended k_min using agent's problem complexity measure
        problem_complexity = self.agent.get_problem_complexity(calibration_states[0])
        k_min = self.harness.calculate_k_min(p_estimate, problem_complexity)

        calibration_results = {
            'p_estimate': p_estimate,
            'k_min': k_min,
            'successful_steps': int(p_estimate * len(calibration_states)),
            'total_steps': len(calibration_states),
            'model': self.config.model,
            'current_k_margin': self.config.k_margin
        }

        logger.info(f"Calibration complete: p={p_estimate:.4f}, k_min={k_min}")
        logger.info(f"Current k_margin={self.config.k_margin}, recommended k_min={k_min}")

        return calibration_results

    def update_k_margin(self, new_k_margin: int):
        """
        Update the k_margin parameter for subsequent executions.

        Args:
            new_k_margin: New k_margin value to use
        """
        old_k = self.config.k_margin
        self.config.k_margin = new_k_margin
        logger.info(f"Updated k_margin: {old_k} -> {new_k_margin}")

    @property
    def total_cost(self) -> float:
        """Get total cost of all API calls"""
        return self.harness.total_cost

    @property
    def total_api_calls(self) -> int:
        """Get total number of API calls made"""
        return self.harness.total_api_calls

    @property
    def total_input_tokens(self) -> int:
        """Get total input tokens used"""
        return self.harness.total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Get total output tokens generated"""
        return self.harness.total_output_tokens

    def get_statistics(self) -> dict:
        """
        Get comprehensive execution statistics.

        Returns:
            Dictionary with execution statistics including costs, tokens, and API calls
        """
        return {
            'total_cost': self.total_cost,
            'total_api_calls': self.total_api_calls,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'model': self.config.model,
            'k_margin': self.config.k_margin,
            'max_candidates': self.config.max_candidates,
            'temperature': self.config.temperature
        }

    def reset_statistics(self):
        """Reset all execution statistics to zero"""
        self.harness.total_cost = 0.0
        self.harness.total_input_tokens = 0
        self.harness.total_output_tokens = 0
        self.harness.total_api_calls = 0
        logger.info("Reset all execution statistics")


async def execute_agent(agent: MicroAgent,
                       *args,
                       config: Optional[MDAPConfig] = None,
                       **kwargs) -> List[Any]:
    """
    Convenience function to execute a microagent in a single call.

    Args:
        agent: The MicroAgent instance to execute
        *args: Arguments to pass to agent.create_initial_state()
        config: Optional MDAPConfig to use
        **kwargs: Additional arguments to pass to agent.create_initial_state()

    Returns:
        List of states representing the execution trace

    Example:
        from mdap.hanoi_solver import HanoiMDAP
        from mdap.micro_agent_executor import execute_agent

        agent = HanoiMDAP()
        trace = await execute_agent(agent, num_disks=5)
        print(f"Solved in {len(trace)-1} steps")
    """
    executor = MicroAgentExecutor(agent, config)
    return await executor.execute(*args, **kwargs)
