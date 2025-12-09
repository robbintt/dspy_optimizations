"""
Simple example demonstrating MDAP Hanoi solver usage

To run this example:
1. Activate the virtual environment: source ~/virtualenvs/mdap_harness_venv/bin/activate
2. Set your API key in .env file
3. Run: python example_hanoi.py
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from hanoi_solver import HanoiMDAP, MDAPConfig

# Setup logging to file with timestamps
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOGS_DIR, f"example_hanoi_{timestamp}.log")

# Configure file handler for example logs
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Create a specific logger for example_hanoi to avoid conflicts
example_logger = logging.getLogger('example_hanoi')
example_logger.setLevel(logging.INFO)
example_logger.addHandler(file_handler)

# Also add console handler to tee output to terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
example_logger.addHandler(console_handler)

# Create logger
logger = logging.getLogger('example_hanoi')

async def main():
    """Simple demonstration of solving Towers of Hanoi"""
    
    print("Starting Hanoi solver example...")
    logger.info("Starting Hanoi solver example")
    logger.info("🏗️  MDAP Hanoi Solver Demo")
    logger.info("=" * 40)
    
    # Parse command line arguments
    num_disks = 2  # default
    k_margin = 6  # set to 6 as requested
    
    if len(sys.argv) > 1:
        try:
            num_disks = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of disks: {sys.argv[1]}")
            return
    
    # Check for --k flag
    if "--k" in sys.argv:
        try:
            k_index = sys.argv.index("--k")
            if k_index + 1 < len(sys.argv):
                k_margin = int(sys.argv[k_index + 1])
            else:
                print("--k flag requires a value")
                return
        except ValueError:
            print("Invalid k value")
            return
    
    # Create solver with default config (uses MDAP_DEFAULT_MODEL from env or default)
    config = MDAPConfig()
    # Override k_margin if provided
    if k_margin is not None:
        config.k_margin = k_margin
    # Enable mock mode if environment variable is set
    if os.getenv("MDAP_MOCK_MODE", "false").lower() == "true":
        config.mock_mode = True
        config.k_margin = 1  # Lower K for faster testing
        config.max_candidates = 2  # Fewer candidates for faster testing
    logger.info(f"Created solver with config: model={config.model}, k_margin={config.k_margin}, mock_mode={config.mock_mode}")
    print(f"Config: model={config.model}, k_margin={config.k_margin}, mock_mode={config.mock_mode}")
    solver = HanoiMDAP(config)
    
    try:
        # Solve Hanoi with specified number of disks
        logger.info(f"Attempting to solve {num_disks}-disk Towers of Hanoi")
        logger.info(f"Solving {num_disks}-disk Towers of Hanoi...")
        print("About to call solve_hanoi...")
        trace = await solver.solve_hanoi(num_disks)
        print("solve_hanoi completed")
        
        # Print solution summary
        final_state = trace[-1]
        optimal_moves = 2**num_disks - 1
        logger.info(f"Solution completed in {final_state.move_count} moves")
        logger.info(f"Solution trace: {len(trace)} states")
        logger.info(f"\n✅ Solved in {final_state.move_count} moves!")
        logger.info(f"Optimal solution: {optimal_moves} moves")
        
        if final_state.move_count == optimal_moves:
            logger.info("Found optimal solution")
            logger.info("🎯 Found optimal solution!")
        else:
            logger.info(f"Used {final_state.move_count - optimal_moves} extra moves beyond optimal")
            logger.info(f"Used {final_state.move_count - optimal_moves} extra moves")
        
        # Show initial and final states
        logger.info(f"\nInitial state: {trace[0].pegs}")
        logger.info(f"Final state:   {trace[-1].pegs}")
        
    except Exception as e:
        logger.error(f"Failed to solve Hanoi: {e}")
        logger.error("Possible causes: API key issue, model unavailable, network problems, rate limiting")
        logger.error(f"\n❌ Failed to solve Hanoi: {e}")
        logger.error("This could be due to:")
        logger.error("  - API key not set or invalid")
        logger.error("  - Model not available or responding with None")
        logger.error("  - Network connectivity issues")
        logger.error("  - Rate limiting or quota exceeded")
        logger.error("\nPlease check your .env file and try again.")
        return

if __name__ == "__main__":
    try:
        print("About to run main()...")
        asyncio.run(main())
        print("Main() completed successfully")
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()
