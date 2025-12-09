import asyncio
import argparse
import os
import sys
from datetime import datetime

# Add the project root to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hanoi_solver import HanoiMDAP
from mdap_harness import MDAPConfig

# Setup logging to file with timestamps
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOGS_DIR, f"calibrate_hanoi_{timestamp}.log")

# Configure file handler for calibration logs
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Create a specific logger for calibrate_hanoi to avoid conflicts
calibrate_logger = logging.getLogger('calibrate_hanoi')
calibrate_logger.setLevel(logging.INFO)
calibrate_logger.addHandler(file_handler)

# Also add console handler to tee output to terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
calibrate_logger.addHandler(console_handler)

async def main():
    logger.info("Starting Hanoi MDAP calibration")
    parser = argparse.ArgumentParser(description="Calibrate k_margin for the Hanoi MDAP solver.")
    parser.add_argument("num_disks", type=int, help="Number of disks for the Hanoi problem to calibrate on.")
    parser.add_argument("--model", type=str, default=os.getenv("MDAP_DEFAULT_MODEL", "cerebras/zai-glm-4.6"), help="The model to calibrate.")
    parser.add_argument("--sample_steps", type=int, default=10, help="Number of steps to use for estimating the per-step success rate.")
    parser.add_argument("--target_reliability", type=float, default=0.95, help="Target reliability (t) for the calculation.")
    
    args = parser.parse_args()

    logger.info(f"Calibration parameters:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Problem: {args.num_disks}-disk Towers of Hanoi")
    logger.info(f"  Sample Steps: {args.sample_steps}")
    logger.info(f"  Target Reliability: {args.target_reliability}")

    logger.info(f"🔧 Calibrating k_margin for model: {args.model}")
    logger.info(f"   Problem: {args.num_disks}-disk Towers of Hanoi")
    logger.info(f"   Sample Steps: {args.sample_steps}")
    logger.info(f"   Target Reliability: {args.target_reliability}")
    logger.info("-" * 20)

    # Use a temporary config for calibration
    config = MDAPConfig(model=args.model, k_margin=3) # k_margin doesn't matter for estimation
    solver = HanoiMDAP(config=config)
    
    # 1. Estimate per-step success rate
    logger.info("Step 1: Estimating per-step success rate")
    p_estimate = await solver.harness.estimate_per_step_success_rate(solver, args.num_disks, args.sample_steps)

    if p_estimate == 0:
        logger.error("Calibration failed: Model could not solve any of the sample steps")
        logger.error("❌ Calibration failed: Model could not solve any of the sample steps.")
        logger.error("   This model may not be suitable for this task.")
        return

    logger.info(f"Estimated per-step success rate: {p_estimate:.4f}")
    
    # 2. Calculate the optimal k_margin
    logger.info("Step 2: Calculating optimal k_margin")
    k_min = solver.harness.calculate_k_min(p_estimate, args.num_disks, args.target_reliability)
    
    logger.info("Calibration completed successfully")
    logger.info(f"Results: p={p_estimate:.4f}, k_margin={k_min}")
    
    logger.info("\n--- Calibration Result ---")
    logger.info(f"Estimated per-step success rate (p): {p_estimate:.4f}")
    logger.info(f"Recommended k_margin: {k_min}")
    logger.info("\nTo use this value, run the solver with the environment variable set:")
    logger.info(f"export MDAP_K_MARGIN={k_min}")
    logger.info(f"./run_mdap.sh example {args.num_disks}")


if __name__ == "__main__":
    asyncio.run(main())
