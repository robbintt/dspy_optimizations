import asyncio
import argparse
import logging
import os
import sys
import random
import pickle
import copy
import yaml
from datetime import datetime
from typing import List

# Add the project root to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdap.hanoi_solver import HanoiMDAP, HanoiState
from mdap.mdap_harness import MDAPConfig

def load_model_config(config_path: str = "config/models.yaml") -> dict:
    """Load model configuration from YAML file"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, config_path)
    
    with open(full_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Setup logging to file with timestamps
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOGS_DIR, f"calibrate_hanoi_{timestamp}.log")

# Create a specific logger for calibrate_hanoi to avoid conflicts
logger = logging.getLogger('calibrate_hanoi')
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger

# Clear any existing handlers to avoid duplicates
if logger.handlers:
    logger.handlers.clear()

# Configure formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Configure file handler for calibration logs
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Also add console handler to tee output to terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def generate_hanoi_solution(num_disks: int):
    """Generate the complete optimal solution for Towers of Hanoi using recursion"""
    def hanoi(n, source, target, auxiliary, moves):
        """Recursive helper to generate moves"""
        if n == 1:
            moves.append([1, source, target])
        else:
            hanoi(n-1, source, auxiliary, target, moves)
            moves.append([n, source, target])
            hanoi(n-1, auxiliary, target, source, moves)
    
    moves = []
    hanoi(num_disks, 0, 2, 1, moves)
    return moves

def get_sample_size(num_disks: int) -> int:
    """
    Return appropriate sample size based on problem complexity.
    Following the paper's approach: larger samples for bigger problems.
    """
    if num_disks <= 5:
        return min(10, 2**num_disks - 1)
    elif num_disks <= 10:
        return 20
    else:
        return min(50, 2**num_disks - 1)  # Cap at 50 samples

def sample_states_evenly(states: list, sample_size: int) -> list:
    """
    Sample states evenly across the solution to get diverse difficulty levels.
    This follows the paper's approach of sampling across the full solution space.
    """
    if len(states) <= sample_size:
        return states
    
    # Sample evenly across the solution: beginning, middle, end
    indices = []
    step_size = len(states) / (sample_size + 1)
    for i in range(sample_size):
        idx = min(int((i + 1) * step_size), len(states) - 1)
        indices.append(idx)
    
    return [states[i] for i in indices]

def apply_moves_to_states(num_disks: int, moves: List[List[int]]):
    """Apply a sequence of moves to generate all states efficiently"""
    states = []
    
    # Use tuple representation for pegs for faster copying
    pegs_tuple = (tuple(range(num_disks, 0, -1)), (), ())  # (A, B, C)
    
    # Initial state - convert tuple to dict for HanoiState
    initial_state = HanoiState(
        pegs={'A': list(pegs_tuple[0]), 'B': list(pegs_tuple[1]), 'C': list(pegs_tuple[2])},
        num_disks=num_disks,
        move_history=[]
    )
    states.append(initial_state)
    
    # Apply each move using tuple operations for speed
    for i, move in enumerate(moves):
        disk_id, from_peg, to_peg = move
        
        # Convert current state to tuples for manipulation
        current_pegs = (
            tuple(states[-1].pegs['A']),
            tuple(states[-1].pegs['B']),
            tuple(states[-1].pegs['C'])
        )
        
        # Apply move to tuples (faster than list operations)
        from_list = list(current_pegs[from_peg])
        disk = from_list.pop()
        to_list = list(current_pegs[to_peg])
        to_list.append(disk)
        
        # Create new pegs tuple
        new_pegs_list = [list(p) for p in current_pegs]
        new_pegs_list[from_peg] = from_list
        new_pegs_list[to_peg] = to_list
        new_pegs_tuple = tuple(new_pegs_list)
        
        # Create new state with minimal copying
        # Store just the previous move for prompt generation
        previous_move = move if i > 0 else None
        new_state = HanoiState(
            pegs={'A': list(new_pegs_tuple[0]), 'B': list(new_pegs_tuple[1]), 'C': list(new_pegs_tuple[2])},
            num_disks=num_disks,
            move_count=i + 1,
            move_history=None,  # Don't store full history
            previous_move=previous_move  # Just store the previous move
        )
        
        states.append(new_state)
        
        # Progress counter
        if (i + 1) % 10000 == 0:
            print(f"Progress: {i + 1:,} states generated...")
    
    return states

async def generate_calibration_cache(num_disks: int = 20, cache_file: str = "calibration_cache.pkl"):
    """Generate and cache calibration states from a full disc solution"""
    logger.info(f"Generating calibration cache for {num_disks}-disc Hanoi problem")
    
    # Suppress INFO level logging during cache generation to avoid massive output
    logging.getLogger('mdap_harness').setLevel(logging.WARNING)
    logging.getLogger('hanoi_solver').setLevel(logging.WARNING)
    
    # Generate the optimal solution directly
    logger.info("Generating optimal moves...")
    print(f"Generating optimal solution for {num_disks} disks (expected {2**num_disks - 1:,} moves)...")
    moves = generate_hanoi_solution(num_disks)
    
    # Apply moves to generate all states
    logger.info("Applying moves to generate states...")
    print(f"Applying {len(moves):,} moves to generate states...")
    full_solution = apply_moves_to_states(num_disks, moves)
    
    # Sample up to 100,000 states for faster calibration
    max_samples = min(100000, len(full_solution))
    # Always include the first and last states
    middle_states = full_solution[1:-1]
    num_middle_samples = min(max_samples - 2, len(middle_states))
    sampled_middle = random.sample(middle_states, num_middle_samples) if middle_states else []
    sampled_states = [full_solution[0]] + sampled_middle + [full_solution[-1]]
    
    # Cache the sampled states
    calibration_data = {
        'num_disks': num_disks,
        'states': sampled_states,
        'total_steps': len(full_solution) - 1,  # Subtract 1 to exclude initial state (moves count)
        'sampled_count': len(sampled_states)
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(calibration_data, f)
    
    # Restore original logging levels
    logging.getLogger('mdap_harness').setLevel(logging.INFO)
    logging.getLogger('hanoi_solver').setLevel(logging.INFO)
    
    logger.info(f"Cached {len(sampled_states)} calibration states to {cache_file}")
    logger.info(f"Full solution has {len(full_solution)} steps")
    
    return calibration_data

async def main():
    logger.info("Starting Hanoi MDAP calibration")
    
    # Load model configuration
    model_config = load_model_config()
    provider = model_config['model']['provider']
    model_name = model_config['model']['name']
    default_model = f"{provider}/{model_name}"
    
    parser = argparse.ArgumentParser(description="Calibrate k_margin for the Hanoi MDAP solver.")
    parser.add_argument("--model", type=str, default=os.getenv("MDAP_DEFAULT_MODEL", default_model), help="The model to calibrate.")
    parser.add_argument("--sample_steps", type=int, default=20, help="Number of steps to use for estimating the per-step success rate.")
    parser.add_argument("--target_reliability", type=float, default=0.95, help="Target reliability (t) for the calculation.")
    parser.add_argument("--cache_file", type=str, default="calibration_cache.pkl", help="Path to calibration cache file.")
    parser.add_argument("--target_disks", type=int, default=20, help="Number of disks in the final target problem for k_margin calculation.")
    parser.add_argument("--use_cache", action="store_true", help="Use an existing calibration cache if available. Default is to regenerate.")
    
    args = parser.parse_args()

    logger.info(f"Calibration parameters:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Problem: 20-disk Towers of Hanoi (following paper)")
    logger.info(f"  Target Problem Size for k_margin: {args.target_disks} disks")
    logger.info(f"  Sample Steps: {args.sample_steps}")
    logger.info(f"  Target Reliability: {args.target_reliability}")
    logger.info(f"  Cache File: {args.cache_file}")

    logger.info(f"🔧 Calibrating k_margin for model: {args.model}")
    logger.info(f"   Problem: 20-disk Towers of Hanoi (following paper)")
    logger.info(f"   Sample Steps: {args.sample_steps}")
    logger.info(f"   Target Reliability: {args.target_reliability}")
    logger.info("-" * 20)

    # Generate or load calibration cache
    # Default behavior is to regenerate unless --use_cache is specified and the file exists.
    if not args.use_cache or not os.path.exists(args.cache_file):
        logger.info("Regenerating calibration cache (default behavior). Use --use_cache to skip if a cache exists.")
        calibration_data = await generate_calibration_cache(cache_file=args.cache_file)
    else:
        with open(args.cache_file, 'rb') as f:
            calibration_data = pickle.load(f)
        logger.info(f"Using existing calibration cache with {calibration_data['sampled_count']} states")

    # Use a temporary config for calibration with lower k_margin for testing
    # Pass the full model configuration
    config = MDAPConfig(
        model=args.model, 
        k_margin=1,  # Use k_margin=1 for calibration to avoid overconfidence
        temperature=model_config['model']['temperature'],
        max_tokens=model_config['model']['max_tokens'],
        top_p=model_config['model']['top_p'],
        frequency_penalty=model_config['model']['frequency_penalty'],
        presence_penalty=model_config['model']['presence_penalty'],
        disable_reasoning=model_config['model'].get('disable_reasoning'),
        reasoning_effort=model_config['model'].get('reasoning_effort'),
        thinking_budget=model_config['model']['thinking_budget'],
        cost_per_input_token=model_config['model']['cost_per_input_token'],
        cost_per_output_token=model_config['model']['cost_per_output_token'],
        max_response_length=model_config['model']['max_response_length'],
        enable_harness_logging=False  # Disable separate harness log file
    )
    solver = HanoiMDAP(config=config)
    
    # 1. Estimate per-step success rate using random subset
    logger.info("Step 1: Estimating per-step success rate")
    
    # Determine appropriate sample size based on problem complexity
    # Following the paper's approach: larger samples for bigger problems
    sample_size = get_sample_size(args.sample_steps)
    
    # For small sample sizes, generate states from appropriately sized problems
    if sample_size <= 20:
        # Use at least 3 disks for calibration to avoid trivial edge cases
        disk_count = max(3, min(5, sample_size))
        logger.info(f"Generating fresh states for {disk_count}-disk calibration (sample_size={sample_size})")
        moves = generate_hanoi_solution(disk_count)
        full_solution = apply_moves_to_states(disk_count, moves)
        # Sample states evenly across the solution to get diverse difficulty levels
        calibration_states = sample_states_evenly(full_solution, sample_size)
    else:
        # Use cached 20-disk states for larger calibrations
        # Sample evenly across the full solution space
        calibration_states = sample_states_evenly(calibration_data['states'], sample_size)
    
    p_estimate = await solver.harness.estimate_per_step_success_rate_from_states(
        solver, calibration_states)

    if p_estimate == 0:
        logger.error("Calibration failed: Model could not solve any of the sample steps")
        logger.error("❌ Calibration failed: Model could not solve any of the sample steps.")
        logger.error("   This model may not be suitable for this task.")
        return

    logger.info(f"Estimated per-step success rate: {p_estimate:.4f}")
    
    # 2. Calculate the optimal k_margin
    logger.info(f"Step 2: Calculating optimal k_margin for target problem ({args.target_disks} disks)")
    k_min = solver.harness.calculate_k_min(p_estimate, args.target_disks, args.target_reliability)
    
    # Add confidence interval estimation for p
    if sample_size > 0:
        import math
        # Wilson score interval for binomial proportion
        z = 1.96  # 95% confidence
        n = sample_size
        p_hat = p_estimate
        denominator = 1 + z**2/n
        centre_adjusted = p_hat + z**2/(2*n)
        margin = z * math.sqrt((p_hat*(1-p_hat) + z**2/(4*n))/n)
        
        ci_lower = (centre_adjusted - margin) / denominator
        ci_upper = (centre_adjusted + margin) / denominator
        
        logger.info(f"95% confidence interval for p: [{ci_lower:.4f}, {ci_upper:.4f}")
        
        # Early stopping warning if p is too low
        if p_estimate < 0.3:
            logger.error(f"🚨 CRITICAL: Model performance dropped to {p_estimate:.1%}")
            logger.error(f"   This model may not be suitable for problems larger than {disk_count_for_k-1} disks")
            logger.error(f"   Consider using a more capable model or reducing problem complexity")
    
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
