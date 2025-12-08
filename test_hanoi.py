"""
Test script for MDAP Hanoi solver
"""

import asyncio
import sys
from hanoi_solver import HanoiMDAP, MDAPConfig

async def test_hanoi_solver():
    """Test the Hanoi solver with different disk counts"""
    
    # Test configurations
    test_cases = [
        {"disks": 3, "expected_moves": 7},
        {"disks": 4, "expected_moves": 15},
    ]
    
    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing Towers of Hanoi with {case['disks']} disks")
        print(f"Expected optimal moves: {case['expected_moves']}")
        print(f"{'='*50}")
        
        # Configure MDAP for this test
        config = MDAPConfig(
            model="gpt-4o-mini",
            k_margin=3,
            max_candidates=8,
            temperature=0.1,
            max_retries=3
        )
        
        solver = HanoiMDAP(config)
        
        try:
            # Solve the puzzle
            trace = await solver.solve_hanoi(case['disks'])
            
            # Verify solution
            final_state = trace[-1]
            if solver.is_solved(final_state):
                print(f"✅ SUCCESS: Solved in {final_state.move_count} moves")
                if final_state.move_count == case['expected_moves']:
                    print("✅ OPTIMAL: Found optimal solution!")
                else:
                    print(f"⚠️  SUBOPTIMAL: Used {final_state.move_count - case['expected_moves']} extra moves")
                
                # Print first few and last few moves
                print("\nFirst 3 moves:")
                for i in range(min(3, len(trace))):
                    state = trace[i]
                    print(f"  Step {i}: A={state.pegs['A']}, B={state.pegs['B']}, C={state.pegs['C']}")
                
                if len(trace) > 6:
                    print("...")
                    print("Last 3 moves:")
                    for i in range(len(trace)-3, len(trace)):
                        state = trace[i]
                        print(f"  Step {i}: A={state.pegs['A']}, B={state.pegs['B']}, C={state.pegs['C']}")
                
            else:
                print("❌ FAILED: Solution is invalid")
                return False
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return False
    
    return True

async def benchmark_performance():
    """Benchmark performance with different K margins"""
    print(f"\n{'='*50}")
    print("BENCHMARK: Testing different K margins")
    print(f"{'='*50}")
    
    k_values = [1, 2, 3, 4]
    disks = 3
    
    for k in k_values:
        print(f"\nTesting K={k}:")
        config = MDAPConfig(
            model="gpt-4o-mini",
            k_margin=k,
            max_candidates=10,
            temperature=0.1
        )
        
        solver = HanoiMDAP(config)
        
        try:
            trace = await solver.solve_hanoi(disks)
            moves = trace[-1].move_count
            print(f"  ✅ K={k}: {moves} moves")
        except Exception as e:
            print(f"  ❌ K={k}: Failed - {e}")

if __name__ == "__main__":
    print("MDAP Hanoi Solver Test Suite")
    print("=" * 50)
    
    # Run tests
    success = asyncio.run(test_hanoi_solver())
    
    if success:
        print("\n🎉 All tests passed! Running benchmark...")
        asyncio.run(benchmark_performance())
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)

# Add pytest test functions
def test_hanoi_solver_sync():
    """Synchronous wrapper for pytest"""
    return asyncio.run(test_hanoi_solver())

def test_benchmark_performance_sync():
    """Synchronous wrapper for pytest"""
    return asyncio.run(benchmark_performance())
