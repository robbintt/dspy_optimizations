"""
Protocol interfaces for the microagent library.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Callable, Tuple

class ExecutionHarness(ABC):
    """
    Abstract protocol for an execution harness that runs a MicroAgent.

    A harness is responsible for the low-level execution, including LLM interaction,
    retries, voting, and state progression. This protocol decouples the
    microagent library from any specific harness implementation (e.g., MDAP).
    """

    @abstractmethod
    async def execute_step(self,
                          step_prompt: Tuple[str, str],
                          response_parser: Callable[[str], Any]) -> Any:
        """
        Execute a single atomic step with error correction.

        Args:
            step_prompt: A tuple containing (system_prompt, user_prompt).
            response_parser: A callable to parse the LLM's raw response.

        Returns:
            The parsed result of the step.
        """
        pass

    @abstractmethod
    async def execute_plan(self,
                          initial_state: Any,
                          step_generator: Callable[[Any], Tuple[Tuple[str, str], Callable[[str], Any]]],
                          termination_check: Callable[[Any], bool],
                          agent: 'MicroAgent') -> List[Any]:
        """
        Execute a complete plan using a micro agent.

        Args:
            initial_state: The starting state of the problem.
            step_generator: A callable from the agent that provides (prompt, parser) for each step.
            termination_check: A callable from the agent to check if the problem is solved.
            agent: The micro agent instance being executed.

        Returns:
            A list of states representing the execution trace.
        """
        pass
