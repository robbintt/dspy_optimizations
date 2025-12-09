"""
Simple example demonstrating MDAP Hanoi solver usage

To run this example:
1. Activate the virtual environment: source ~/virtualenvs/mdap_harness_venv/bin/activate
2. Set your API key in .env file
3. Run: python example_hanoi.py
"""

import asyncio
import os
from datetime import datetime
from hanoi_solver import HanoiMDAP, MDAPConfig

# Setup logging to file with timestamps
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOGS_DIR, f"example_hanoi_{timestamp}.log")

# Configure file handler for example logs
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to root logger
logging.getLogger().addHandler(file_handler)

# Also add console handler to tee output to terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

async def main():
    """Simple demonstration of solving Towers of Hanoi"""
    
    logger.info("Starting Hanoi solver example")
    logger.info("🏗️  MDAP Hanoi Solver Demo")
    logger.info("=" * 40)
    
    # Create solver with default config (uses MDAP_DEFAULT_MODEL from env or default)
    config = MDAPConfig()
    logger.info(f"Created solver with config: model={config.model}, k_margin={config.k_margin}")
    solver = HanoiMDAP(config)
    
    try:
        # Solve 2-disk Hanoi
        logger.info("Attempting to solve 2-disk Towers of Hanoi")
        logger.info("Solving 2-disk Towers of Hanoi...")
        trace = await solver.solve_hanoi(2)
        
        # Print solution summary
        final_state = trace[-1]
        logger.info(f"Solution completed in {final_state.move_count} moves")
        logger.info(f"Solution trace: {len(trace)} states")
        logger.info(f"\n✅ Solved in {final_state.move_count} moves!")
        logger.info(f"Optimal solution: {2**2 - 1} moves")
        
        if final_state.move_count == 3:
            logger.info("Found optimal solution")
            logger.info("🎯 Found optimal solution!")
        else:
            logger.info(f"Used {final_state.move_count - 3} extra moves beyond optimal")
            logger.info(f"Used {final_state.move_count - 3} extra moves")
        
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
    asyncio.run(main())
