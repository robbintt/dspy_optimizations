import asyncio
import logging
import os

from .agent import HelloWorldAgent
from microagent import MicroAgentExecutor, LiteLLMConfig

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
import yaml
import os
from typing import Any, List, Callable, Tuple
from .agent import HelloWorldAgent
from microagent.protocols import ExecutionHarness
from microagent import MicroAgent, LiteLLMHarness

logger = logging.getLogger(__name__)

class HelloWorldHarness(ExecutionHarness):
    """
    An execution harness for the HelloWorldAgent that uses the LiteLLMHarness.
    It loads model configuration from a local YAML file.
    """

    def __init__(self, agent: HelloWorldAgent):
        self.agent = agent
        
        # Define the default path to the configuration file
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'model.yml')

        # Create the configuration object from the file
        llm_config = LiteLLMConfig(config_file=config_path)

        # Instantiate the underlying LiteLLMHarness with the config object
        self._harness = LiteLLMHarness(config=llm_config)
        logger.info(f"Initialized {self.__class__.__name__} with model: {llm_config.model}")

    @property
    def total_cost(self):
        return self._harness.total_cost

    @property
    def total_api_calls(self):
        return self._harness.total_api_calls
    
    async def execute_step(self,
                           step_prompt: Tuple[str, str],
                           response_parser: Callable[[str], Any]) -> Any:
        """Delegate step execution to the internal LiteLLMHarness."""
        return await self._harness.execute_step(step_prompt, response_parser)

    async def execute_plan(self,
                          initial_state: Any,
                          step_generator: Callable[[Any], Tuple[str, Callable]],
                          termination_check: Callable[[Any], bool],
                          agent: MicroAgent) -> List[Any]:
        """
        Adapt the agent's simple prompt to a (system, user) tuple for LiteLLM,
        then delegate execution to the internal LiteLLMHarness.
        """
        def adapted_step_generator(state):
            prompt, parser = agent.step_generator(state)
            
            system_prompt = "You are an assistant that follows instructions perfectly and responds with only the requested content."
            user_prompt = prompt

            return (system_prompt, user_prompt), parser

        return await self._harness.execute_plan(
            initial_state=initial_state,
            step_generator=adapted_step_generator,
            termination_check=termination_check,
            agent=agent
        )
