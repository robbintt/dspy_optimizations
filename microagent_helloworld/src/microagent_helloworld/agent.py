import logging
import re
from microagent import MicroAgent

logger = logging.getLogger(__name__)

class HelloWorldAgent(MicroAgent):
    """A simple agent that builds a target string character by character."""

    TARGET_STRING = "Hello, World!"

    def create_initial_state(self) -> str:
        """Start with an empty string."""
        return ""

    def generate_step_prompt(self, state: str) -> str:
        """Generate a prompt that asks the LLM for the next character."""
        if len(state) >= len(self.TARGET_STRING):
            return "The process is already complete."

        return (
            f"The goal is to build the exact string: '{self.TARGET_STRING}'.\n"
            f"The current string is: '{state}'.\n"
            "Provide the single next character to append. "
            "Respond with only that character and nothing else."
        )

    def get_response_parser(self):
        """Parse the LLM's response to extract a single character."""
        def parser(response: str) -> str:
            match = re.search(r"\S", response)
            return match.group(0) if match else ""
        return parser

    def validate_step_result(self, step_result: str) -> bool:
        """Validate that the result from the parser is a single character."""
        return isinstance(step_result, str) and len(step_result) == 1

    def update_state(self, current_state: str, step_result: str) -> str:
        """Append the new character to the current state string."""
        logger.info(f"Appending '{step_result}'. State: '{current_state}' -> '{current_state + step_result}'")
        return current_state + step_result

    def is_solved(self, state: str) -> bool:
        """Check if the target string has been successfully built."""
        solved = state == self.TARGET_STRING
        if solved:
            logger.info(f"Success! Agent built the target string: '{state}'")
        return solved

    def get_problem_complexity(self, state: str) -> int:
        """The complexity is the total length of the target string."""
        return len(self.TARGET_STRING)
