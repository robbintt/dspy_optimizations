import asyncio
import logging
import os

from .agent import HelloWorldAgent
from microagent import MicroAgentExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def run():
    """
    Main function to run the HelloWorldAgent demo.
    
    Execution requires a configured LLM client, e.g., by setting an
    environment variable such as `OPENAI_API_KEY`.
    """
    print("--- Starting MicroAgent HelloWorld Demo ---")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: 'OPENAI_API_KEY' not found in environment. "
              "Execution will likely fail.")

    agent = HelloWorldAgent()
    
    # Create and use the local HelloWorldHarness
    from .harness import HelloWorldHarness
    harness = HelloWorldHarness(agent=agent)
    executor = MicroAgentExecutor(agent, harness)

    print(f"Target String: '{agent.TARGET_STRING}'\n")
    print(f"Using harness: {harness.__class__.__name__}\n")
    try:
        trace = await executor.execute()
        final_state = trace[-1]

        print("\n--- Execution Complete ---")
        print(f"Final State: '{final_state}'")
        if final_state == agent.TARGET_STRING:
            print("✅ Success! The agent built the string correctly.")
        else:
            print(f"❌ Failure. The agent did not build the string correctly.")

        print("\n--- Execution Statistics ---")
        stats = {
            'Total Cost': f"${harness.total_cost:.4f}",
            'Total API Calls': harness.total_api_calls
        }
        for key, value in stats.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"\n❌ An error occurred during execution: {e}")
        print("Please ensure your LLM client is configured and accessible.")

def main():
    """Synchronous entry point for the script defined in pyproject.toml."""
    asyncio.run(run())

if __name__ == "__main__":
    main()
import logging
from typing import Any, List, Callable, Tuple
from .agent import HelloWorldAgent
from microagent.protocols import ExecutionHarness
from microagent import MicroAgent

logger = logging.getLogger(__name__)

class HelloWorldHarness(ExecutionHarness):
    """
    A simple execution harness for the HelloWorldAgent.
    This harness simulates an LLM call by providing the correct next
    character in the target string to demonstrate the framework's logic.
    """

    def __init__(self, agent: HelloWorldAgent):
        self.agent = agent
        self.total_cost = 0.0
        self.total_api_calls = 0
        logger.info(f"Initialized {self.__class__.__name__}")

    async def execute_step(self,
                           step_prompt: Tuple[str, str],
                           response_parser: Callable[[str], Any]) -> Any:
        """Simulates an LLM call. Returns the correct next character."""
        self.total_api_calls += 1
        
        current_state = step_prompt[1]
        expected_next_char = self.agent.TARGET_STRING[len(current_state)]

        logger.info(f"Simulated LLM response: '{expected_next_char}'")
        return response_parser(expected_next_char)

    async def execute_plan(self,
                          initial_state: Any,
                          step_generator: Callable[[Any], Tuple[Tuple[str, str], Callable]],
                          termination_check: Callable[[Any], bool],
                          agent: MicroAgent) -> List[Any]:
        """Executes the plan step-by-step until terminated."""
        trace = [initial_state]
        current_state = initial_state

        while not termination_check(current_state):
            prompt, parser = step_generator(current_state)
            result = await self.execute_step(prompt, parser)
            
            current_state = agent.update_state(current_state, result)
            trace.append(current_state)
            logger.info(f"State updated to: '{current_state}'")

        self.total_cost = self.total_api_calls * 0.0001
        return trace
