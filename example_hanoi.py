"""
Simple example demonstrating MDAP Hanoi solver usage
"""

import asyncio
from hanoi_solver import HanoiMDAP, MDAPConfig

async def main():
    """Simple demonstration of solving Towers of Hanoi"""
    
    print("🏗️  MDAP Hanoi Solver Demo")
    print("=" * 40)
    
    # Create solver with default config
    solver = HanoiMDAP()
    
    # Solve 3-disk Hanoi
    print("Solving 3-disk Towers of Hanoi...")
    trace = await solver.solve_hanoi(3)
    
    # Print solution summary
    final_state = trace[-1]
    print(f"\n✅ Solved in {final_state.move_count} moves!")
    print(f"Optimal solution: {2**3 - 1} moves")
    
    if final_state.move_count == 7:
        print("🎯 Found optimal solution!")
    else:
        print(f"Used {final_state.move_count - 7} extra moves")
    
    # Show initial and final states
    print(f"\nInitial state: {trace[0].pegs}")
    print(f"Final state:   {trace[-1].pegs}")

if __name__ == "__main__":
    asyncio.run(main())
