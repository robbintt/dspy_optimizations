#!/usr/bin/env python3
"""
Basic test script for MicroAgentExecutor to verify installation and imports.
"""

from micro_agent_executor import MicroAgentExecutor
from hanoi_solver import HanoiMDAP

def main():
    print("Testing MicroAgentExecutor...")
    print("=" * 50)

    # Test imports
    print("✓ Successfully imported MicroAgentExecutor")
    print("✓ Successfully imported HanoiMDAP")

    # Test instantiation
    agent = HanoiMDAP()
    executor = MicroAgentExecutor(agent)
    print("✓ Successfully created executor")

    # Test properties
    print(f"\nConfiguration:")
    print(f"  Model: {executor.config.model}")
    print(f"  k_margin: {executor.config.k_margin}")
    print(f"  max_candidates: {executor.config.max_candidates}")
    print(f"  temperature: {executor.config.temperature}")

    # Test statistics (should be zero initially)
    print(f"\nInitial Statistics:")
    print(f"  Total cost: ${executor.total_cost:.4f}")
    print(f"  Total API calls: {executor.total_api_calls}")
    print(f"  Input tokens: {executor.total_input_tokens}")
    print(f"  Output tokens: {executor.total_output_tokens}")

    stats = executor.get_statistics()
    print(f"\nFull statistics dict has {len(stats)} entries")

    print("\n✓ All basic checks passed!")
    print("=" * 50)
    return 0

if __name__ == "__main__":
    exit(main())
