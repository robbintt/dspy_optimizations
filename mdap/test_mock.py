#!/usr/bin/env python3
"""
Test MDAP with mock mode (no API calls required)
"""

import asyncio
import logging
import os
from datetime import datetime
from hanoi_solver import HanoiMDAP, MDAPConfig

# Setup logging
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOGS_DIR, f"test_mock_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """Test MDAP with mock mode"""
    
    print("Testing MDAP with mock mode...")
    logger.info("Testing MDAP with mock mode")
    
    # Create config with mock mode enabled
    config = MDAPConfig(
        mock_mode=True,
        k_margin=1,  # Lower K for faster testing
        max_candidates=2  # Fewer candidates for faster testing
    )
    
    logger.info(f"Created solver with mock mode enabled")
    solver = HanoiMDAP(config)
    
    try:
        logger.info("Attempting to solve 2-disk Towers of Hanoi with mock mode")
        trace = await solver.solve_hanoi(2)
        
        final_state = trace[-1]
        logger.info(f"Solution completed in {final_state.move_count} moves")
        logger.info(f"Solution trace: {len(trace)} states")
        
        if final_state.move_count == 3:  # Optimal for 2 disks
            logger.info("✅ Found optimal solution!")
        else:
            logger.info(f"Used {final_state.move_count} moves")
        
        print(f"✅ Mock test completed successfully!")
        print(f"Log file: {log_file}")
        
    except Exception as e:
        logger.error(f"Failed to solve Hanoi: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
