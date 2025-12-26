from abc import ABC, abstractmethod
from typing import Any, Tuple, Callable

class MicroAgent(ABC):
    """Abstract base class for micro agents."""
    
    def __init__(self, config=None):
        self.config = config

    @abstractmethod
    def create_initial_state(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def generate_step_prompt(self, state: Any) -> str:
        pass

    @abstractmethod
    def update_state(self, current_state: Any, step_result: Any) -> Any:
        pass

    @abstractmethod
    def is_solved(self, state: Any) -> bool:
        pass
