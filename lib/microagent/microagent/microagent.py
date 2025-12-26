"""
Base class for micro agents in the MDAP framework
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Callable, List, Optional

class MicroAgent(ABC):
    """
    Abstract base class for micro agents that solve specific problems
    using the MDAP (Multi-step Decision and Planning) framework.
    """
    
    def __init__(self, config=None):
        """Initialize the micro agent with optional configuration"""
        self.config = config
    
    @abstractmethod
    def create_initial_state(self, *args, **kwargs) -> Any:
        """Create the initial state for the problem"""
        pass
    
    @abstractmethod
    def generate_step_prompt(self, state: Any) -> str:
        """Generate a prompt for the next step based on current state"""
        pass
    
    @abstractmethod
    def update_state(self, current_state: Any, step_result: Any) -> Any:
        """Update the state based on the step result"""
        pass
    
    @abstractmethod
    def is_solved(self, state: Any) -> bool:
        """Check if the problem is solved"""
        pass

    @abstractmethod
    def get_problem_complexity(self, state: Any) -> int:
        """
        Get the problem complexity for calibration purposes.

        For Hanoi, this would return the number of disks.
        For other problems, return an appropriate measure of problem size/difficulty.

        Returns:
            int: A positive integer representing problem complexity
        """
        pass

    def step_generator(self, state: Any) -> Tuple[str, Callable[[str], Any]]:
        """
        Generate step prompt and parser for the current state.
        Default implementation returns the prompt and a parser that
        subclasses should override if needed.
        """
        prompt = self.generate_step_prompt(state)
        parser = self.get_response_parser()
        return prompt, parser
    
    def get_response_parser(self) -> Callable[[str], Any]:
        """
        Get the response parser for this agent.
        Subclasses should override this to provide domain-specific parsing.
        """
        return lambda x: x  # Default passthrough parser
    
    def validate_step_result(self, step_result: Any) -> bool:
        """
        Validate that a step result is acceptable before updating state.
        Subclasses can override this for domain-specific validation.
        """
        return step_result is not None
