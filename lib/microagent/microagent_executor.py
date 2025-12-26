"""
Generic executor for running MicroAgent implementations.
Provides a clean, reusable interface for executing any MicroAgent-based solver
by injecting an ExecutionHarness implementation.
"""

import logging
from typing import Any, List, Optional
from .microagent import MicroAgent
from .protocols import ExecutionHarness

logger = logging.getLogger(__name__)


class MicroAgentExecutor:
    """
    Generic executor for MicroAgent implementations.

    Provides a unified interface for executing any MicroAgent by injecting an
    ExecutionHarness implementation. This decouples the executor from any
    specific execution strategy.

    Example usage:
        # Create a specific agent implementation and a harness
        agent = HanoiMDAP()
        harness = MDAPHarness(config=harness_config, llm_client=llm_client)

        # Create executor and run
        executor = MicroAgentExecutor(agent, harness)
        result = await executor.execute(num_disks=5)

        # Access execution statistics from the harness
        print(f"Total cost: ${harness.total_cost:.4f}")
    """

    def __init__(self, agent: MicroAgent, harness: ExecutionHarness):
        """
        Initialize the executor with a MicroAgent and an ExecutionHarness.

        Args:
            agent: The MicroAgent instance to execute.
            harness: An object that conforms to the ExecutionHarness protocol.
        """
        self.agent = agent
        self.harness = harness

        logger.info(f"Initialized MicroAgentExecutor with agent: {agent.__class__.__name__}")
        logger.info(f"Using harness: {harness.__class__.__name__}")

    async def execute(self, *args, **kwargs) -> List[Any]:
        """
        Execute the microagent with the given arguments using the injected harness.

        Args:
            *args, **kwargs: Arguments passed to agent.create_initial_state()

        Returns:
            List of states representing the execution trace
        """
        logger.info(f"Starting execution with args={args}, kwargs={kwargs}")

        try:
            # Create the initial state using the agent
            initial_state = self.agent.create_initial_state(*args, **kwargs)

            # Execute the plan using the harness
            trace = await self.harness.execute_plan(
                initial_state=initial_state,
                step_generator=self.agent.step_generator,
                termination_check=self.agent.is_solved,
                agent=self.agent
            )

            logger.info(f"Execution completed successfully")
            logger.info(f"Trace length: {len(trace)} states")

            return trace

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            raise

