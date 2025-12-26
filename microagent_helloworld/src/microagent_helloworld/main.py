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
    executor = MicroAgentExecutor(agent)

    print(f"Target String: '{agent.TARGET_STRING}'\n")
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
        stats = executor.get_statistics()
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

    except Exception as e:
        print(f"\n❌ An error occurred during execution: {e}")
        print("Please ensure your LLM client is configured and accessible.")

def main():
    """Synchronous entry point for the script defined in pyproject.toml."""
    asyncio.run(run())

if __name__ == "__main__":
    main()
