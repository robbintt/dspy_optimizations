"""
Example demonstrating the generic MicroAgentExecutor.
Shows how to use the executor with different agent implementations.
"""

import asyncio
import logging
from hanoi_solver import HanoiMDAP
from microagent_executor import MicroAgentExecutor, execute_agent
from mdap_harness import MDAPConfig

logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Basic usage: create agent and executor, then run"""
    print("\n=== Example 1: Basic Usage ===")

    # Create agent with default config
    agent = HanoiMDAP()

    # Create executor
    executor = MicroAgentExecutor(agent)

    # Execute
    trace = await executor.execute(num_disks=3)

    # Print results
    print(f"Solved {trace[-1].num_disks}-disk Hanoi in {trace[-1].move_count} moves")
    print(f"Total cost: ${executor.total_cost:.4f}")
    print(f"API calls: {executor.total_api_calls}")


async def example_custom_config():
    """Using custom configuration"""
    print("\n=== Example 2: Custom Configuration ===")

    # Create custom config
    config = MDAPConfig(
        k_margin=5,
        max_candidates=15,
        temperature=0.7
    )

    # Create agent
    agent = HanoiMDAP(config)

    # Create executor with custom config
    executor = MicroAgentExecutor(agent, config)

    # Execute
    trace = await executor.execute(num_disks=4)

    # Get detailed statistics
    stats = executor.get_statistics()
    print(f"Execution statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


async def example_convenience_function():
    """Using the convenience function for one-off executions"""
    print("\n=== Example 3: Convenience Function ===")

    # Create agent
    agent = HanoiMDAP()

    # Execute in one line
    trace = await execute_agent(agent, num_disks=3)

    print(f"Solved in {len(trace)-1} steps")


async def example_multiple_runs():
    """Running multiple problems with the same executor"""
    print("\n=== Example 4: Multiple Runs ===")

    agent = HanoiMDAP()
    executor = MicroAgentExecutor(agent)

    for n in [3, 4, 5]:
        # Reset statistics before each run
        executor.reset_statistics()

        trace = await executor.execute(num_disks=n)

        print(f"\n{n} disks:")
        print(f"  Moves: {trace[-1].move_count}")
        print(f"  Cost: ${executor.total_cost:.4f}")
        print(f"  API calls: {executor.total_api_calls}")


async def example_with_calibration():
    """Running calibration before execution"""
    print("\n=== Example 5: With Calibration ===")

    # Note: This example shows the pattern, but requires calibration states
    # In practice, you'd load these from a calibration cache
    agent = HanoiMDAP()
    executor = MicroAgentExecutor(agent)

    # Normally you'd load pre-generated calibration states
    # calibration_states = load_calibration_cache()
    # calibration_results = await executor.calibrate(calibration_states)
    #
    # print(f"Calibration results:")
    # print(f"  Per-step success rate: {calibration_results['p_estimate']:.4f}")
    # print(f"  Recommended k_min: {calibration_results['k_min']}")
    # print(f"  Current k_margin: {calibration_results['current_k_margin']}")
    #
    # # Update k_margin if needed
    # if calibration_results['k_min'] > executor.config.k_margin:
    #     executor.update_k_margin(calibration_results['k_min'])

    # Execute with calibrated settings
    trace = await executor.execute(num_disks=5)
    print(f"Solved in {trace[-1].move_count} moves")


async def main():
    """Run all examples"""
    print("Generic MicroAgent Executor Examples")
    print("=" * 50)

    await example_basic_usage()
    await example_custom_config()
    await example_convenience_function()
    await example_multiple_runs()
    await example_with_calibration()


if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.WARNING,  # Set to INFO to see detailed logs
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run examples
    asyncio.run(main())
