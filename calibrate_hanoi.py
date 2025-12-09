import asyncio
import argparse
import os
import sys

# Add the project root to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hanoi_solver import HanoiMDAP
from mdap_harness import MDAPConfig

async def main():
    parser = argparse.ArgumentParser(description="Calibrate k_margin for the Hanoi MDAP solver.")
    parser.add_argument("num_disks", type=int, help="Number of disks for the Hanoi problem to calibrate on.")
    parser.add_argument("--model", type=str, default=os.getenv("MDAP_DEFAULT_MODEL", "cerebras/zai-glm-4.6"), help="The model to calibrate.")
    parser.add_argument("--sample_steps", type=int, default=10, help="Number of steps to use for estimating the per-step success rate.")
    parser.add_argument("--target_reliability", type=float, default=0.95, help="Target reliability (t) for the calculation.")
    
    args = parser.parse_args()

    print(f"🔧 Calibrating k_margin for model: {args.model}")
    print(f"   Problem: {args.num_disks}-disk Towers of Hanoi")
    print(f"   Sample Steps: {args.sample_steps}")
    print(f"   Target Reliability: {args.target_reliability}")
    print("-" * 20)

    # Use a temporary config for calibration
    config = MDAPConfig(model=args.model, k_margin=3) # k_margin doesn't matter for estimation
    solver = HanoiMDAP(config=config)
    
    # 1. Estimate per-step success rate
    p_estimate = await solver.harness.estimate_per_step_success_rate(solver, args.num_disks, args.sample_steps)

    if p_estimate == 0:
        print("❌ Calibration failed: Model could not solve any of the sample steps.")
        print("   This model may not be suitable for this task.")
        return

    # 2. Calculate the optimal k_margin
    k_min = solver.harness.calculate_k_min(p_estimate, args.num_disks, args.target_reliability)
    
    print("\n--- Calibration Result ---")
    print(f"Estimated per-step success rate (p): {p_estimate:.4f}")
    print(f"Recommended k_margin: {k_min}")
    print("\nTo use this value, run the solver with the environment variable set:")
    print(f"export MDAP_K_MARGIN={k_min}")
    print(f"./run_mdap.sh example {args.num_disks}")


if __name__ == "__main__":
    asyncio.run(main())
