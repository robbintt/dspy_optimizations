import asyncio
import argparse
import logging
import os
import sys
import random
import pickle
import copy
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
logger = logging.getLogger('calibrate_hanoi')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Also add console handler to tee output to terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

async def generate_calibration_cache(num_disks: int = 20, cache_file: str = "calibration_cache.pkl"):
    """Generate and cache calibration states from a full 20-disc solution"""
    logger.info(f"Generating calibration cache for {num_disks}-disc Hanoi problem")
    
    # Generate full solution for 20 discs (2^20 - 1 = 1,048,575 steps)
    config = MDAPConfig(model="cerebras/zai-glm-4.6", k_margin=1)  # Use cerebras model for caching
    solver = HanoiMDAP(config=config)
    
    # Mock the LLM to generate optimal moves
    original_step_generator = solver.step_generator
    
    def mock_step_generator(state):
        """Generate optimal moves without LLM calls"""
        optimal_move = solver.get_optimal_move(state)
        
        # Check if a valid move was found
        if not optimal_move:
            # This should not happen for a valid, unsolved state.
            # Raising an error here gives a much clearer message than the TypeError.
            raise ValueError(f"get_optimal_move returned no valid move for state: {state.to_dict()}")
        
        # Get the previous move from state history if available
        previous_move = "None"
        if hasattr(state, 'move_history') and len(state.move_history) > 0:
            last_move = state.move_history[-1]
            previous_move = f"[{last_move['disk_id']}, {last_move['from_peg']}, {last_move['to_peg']}]"
        
        # Generate the prompt with correct previous move
        prompt = solver.generate_step_prompt(state)
        
        # Generate the predicted state for this move
        # Apply the move directly to get the new state
        disk_id, from_peg, to_peg = optimal_move
        new_pegs = {peg: list(disks) for peg, disks in state.pegs.items()}
        
        # Move the disk - but ensure we're moving from the correct peg
        # The optimal_move uses 0-indexed pegs, but state.pegs uses 'A', 'B', 'C'
        from_peg_char = chr(65 + from_peg)
        to_peg_char = chr(65 + to_peg)
        
        # Validate the move before applying
        if not new_pegs[from_peg_char]:
            raise ValueError(f"Cannot move from empty peg {from_peg_char}")
        
        disk = new_pegs[from_peg_char].pop(0)
        
        # Check if move is valid (can't place larger on smaller)
        if new_pegs[to_peg_char] and disk > new_pegs[to_peg_char][0]:
            raise ValueError(f"Invalid move: placing disk {disk} on top of smaller disk {new_pegs[to_peg_char][0]}")
        
        new_pegs[to_peg_char].insert(0, disk)
        
        # Create new state object with proper move history
        new_state = type(state)(
            pegs=new_pegs,
            num_disks=state.num_disks,
            move_count=state.move_count + 1,
            move_history=copy.deepcopy(state.move_history) if state.move_history else []
        )
        
        # Add current move to history
        new_state.move_history.append({
            'disk_id': disk_id,
            'from_peg': from_peg,
            'to_peg': to_peg
        })
        
        return prompt, lambda x: {
            "move": optimal_move,
            "predicted_state": new_state.to_dict()
        }
    
    solver.step_generator = mock_step_generator
    
    logger.info("Generating full solution...")
    full_solution = await solver.solve_hanoi(num_disks)
    
    # Sample up to 1 million states (the full solution has ~1M steps)
    max_samples = min(1000000, len(full_solution))
    sampled_states = random.sample(full_solution, max_samples)
    
    # Cache the sampled states
    calibration_data = {
        'num_disks': num_disks,
        'states': sampled_states,
        'total_steps': len(full_solution),
        'sampled_count': len(sampled_states)
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(calibration_data, f)
    
    logger.info(f"Cached {len(sampled_states)} calibration states to {cache_file}")
    logger.info(f"Full solution has {len(full_solution)} steps")
    
    return calibration_data

async def main():
    logger.info("Starting Hanoi MDAP calibration")
    parser = argparse.ArgumentParser(description="Calibrate k_margin for the Hanoi MDAP solver.")
    parser.add_argument("--model", type=str, default=os.getenv("MDAP_DEFAULT_MODEL", "cerebras/zai-glm-4.6"), help="The model to calibrate.")
    parser.add_argument("--sample_steps", type=int, default=20, help="Number of steps to use for estimating the per-step success rate.")
    parser.add_argument("--target_reliability", type=float, default=0.95, help="Target reliability (t) for the calculation.")
    parser.add_argument("--cache_file", type=str, default="calibration_cache.pkl", help="Path to calibration cache file.")
    parser.add_argument("--regenerate_cache", action="store_true", help="Regenerate the calibration cache.")
    
    args = parser.parse_args()

    logger.info(f"Calibration parameters:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Problem: 20-disk Towers of Hanoi (following paper)")
    logger.info(f"  Sample Steps: {args.sample_steps}")
    logger.info(f"  Target Reliability: {args.target_reliability}")
    logger.info(f"  Cache File: {args.cache_file}")

    logger.info(f"🔧 Calibrating k_margin for model: {args.model}")
    logger.info(f"   Problem: 20-disk Towers of Hanoi (following paper)")
    logger.info(f"   Sample Steps: {args.sample_steps}")
    logger.info(f"   Target Reliability: {args.target_reliability}")
    logger.info("-" * 20)

    # Generate or load calibration cache
    if args.regenerate_cache or not os.path.exists(args.cache_file):
        calibration_data = await generate_calibration_cache(cache_file=args.cache_file)
    else:
        with open(args.cache_file, 'rb') as f:
            calibration_data = pickle.load(f)
        logger.info(f"Loaded calibration cache with {calibration_data['sampled_count']} states")

    # Use a temporary config for calibration with lower k_margin for testing
    config = MDAPConfig(model=args.model, k_margin=1) # Use k_margin=1 for calibration to avoid overconfidence
    solver = HanoiMDAP(config=config)
    
    # 1. Estimate per-step success rate using random subset
    logger.info("Step 1: Estimating per-step success rate")
    
    # Randomly sample states for calibration
    calibration_states = random.sample(calibration_data['states'], 
                                      min(args.sample_steps, len(calibration_data['states'])))
    
    p_estimate = await solver.harness.estimate_per_step_success_rate_from_states(
        solver, calibration_states)

    if p_estimate == 0:
        logger.error("Calibration failed: Model could not solve any of the sample steps")
        logger.error("❌ Calibration failed: Model could not solve any of the sample steps.")
        logger.error("   This model may not be suitable for this task.")
        return

    logger.info(f"Estimated per-step success rate: {p_estimate:.4f}")
    
    # 2. Calculate the optimal k_margin
    logger.info("Step 2: Calculating optimal k_margin")
    k_min = solver.harness.calculate_k_min(p_estimate, 20, args.target_reliability)
    
    logger.info("Calibration completed successfully")
    logger.info(f"Results: p={p_estimate:.4f}, k_margin={k_min}")
    
    logger.info("\n--- Calibration Result ---")
    logger.info(f"Estimated per-step success rate (p): {p_estimate:.4f}")
    logger.info(f"Recommended k_margin: {k_min}")
    logger.info("\nTo use this value, run the solver with the environment variable set:")
    logger.info(f"export MDAP_K_MARGIN={k_min}")
    logger.info(f"./run_mdap.sh example 20")


if __name__ == "__main__":
    asyncio.run(main())
