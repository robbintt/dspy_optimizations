import asyncio
import logging
import os

from .agent import HelloWorldAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def run():
    """
    Main function to run the HelloWorldAgent demo.

    Execution requires a configured LLM client, e.g., by setting an
    environment variable such as `OPENAI_API_KEY` or `CEREBRAS_API_KEY`.
    """
    print("--- Starting MicroAgent HelloWorld Demo ---")

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("CEREBRAS_API_KEY"):
        print("WARNING: No API key found in environment (OPENAI_API_KEY, CEREBRAS_API_KEY). "
              "Execution will likely fail.")

    # Create the agent - config is loaded automatically from config/model.yml
    agent = HelloWorldAgent()

    print(f"Target String: '{agent.TARGET_STRING}'")
    print(f"Model: {agent.model}\n")

    try:
        # Execute directly on the agent - no separate executor or harness needed
        trace = await agent.execute()
        final_state = trace[-1]

        print("\n--- Execution Complete ---")
        print(f"Final State: '{final_state}'")
        if final_state == agent.TARGET_STRING:
            print("Success! The agent built the string correctly.")
        else:
            print(f"Failure. The agent did not build the string correctly.")

        print("\n--- Execution Statistics ---")
        print(f"Total Cost: ${agent.total_cost:.4f}")
        print(f"Total API Calls: {agent.total_api_calls}")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        print("Please ensure your LLM client is configured and accessible.")


def main():
    """Synchronous entry point for the script defined in pyproject.toml."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
