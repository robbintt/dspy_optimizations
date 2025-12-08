"""
Simple example demonstrating MDAP Hanoi solver usage

To run this example:
1. Activate the virtual environment: source ~/virtualenvs/mdap_harness_venv/bin/activate
2. Set your API key in .env file
3. Run: python example_hanoi.py
"""

import asyncio
from hanoi_solver import HanoiMDAP, MDAPConfig

async def main():
    """Simple demonstration of solving Towers of Hanoi"""
    
    print("🏗️  MDAP Hanoi Solver Demo")
    print("=" * 40)
    
    # Create solver with default config (uses MDAP_DEFAULT_MODEL from env or default)
    config = MDAPConfig()
    solver = HanoiMDAP(config)
    
    # Solve 2-disk Hanoi
    print("Solving 2-disk Towers of Hanoi...")
    trace = await solver.solve_hanoi(2)
    
    # Print solution summary
    final_state = trace[-1]
    print(f"\n✅ Solved in {final_state.move_count} moves!")
    print(f"Optimal solution: {2**2 - 1} moves")
    
    if final_state.move_count == 3:
        print("🎯 Found optimal solution!")
    else:
        print(f"Used {final_state.move_count - 3} extra moves")
    
    # Show initial and final states
    print(f"\nInitial state: {trace[0].pegs}")
    print(f"Final state:   {trace[-1].pegs}")

if __name__ == "__main__":
    asyncio.run(main())
