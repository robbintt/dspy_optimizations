"""
Legacy script for GEPA optimization.
This script is kept for backward compatibility but is deprecated.
Use run_gepa.sh for new optimizations.
"""

import sys
import os
import json
import dspy
from gepa_config import setup_dspy, refinement_gepa_metric, get_default_gepa_run_config, create_gepa_optimizer
from gepa_system import GlmSelfReflect

def main():
    """Run GEPA optimization with default settings."""
    output_file = "glm_gepa_complete.json"
    data_file = "golden_set.json"
    
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found. Please run generate_data.py first.")
        sys.exit(1)
    
    if os.path.exists(output_file):
        print(f"Optimized program already exists at '{output_file}'. Use run_gepa.sh for new optimizations.")
        return
    
    print("Loading data...")
    with open(data_file, "r") as f:
        raw_data = json.load(f)
        trainset = [dspy.Example(**d).with_inputs("question", "draft_answer") for d in raw_data]
        valset = trainset[-5:] 
        trainset = trainset[:-5]
    
    print("Setting up models...")
    task_lm, reflection_lm = setup_dspy()
    
    print("Starting GEPA optimization...")
    gepa_run_config = get_default_gepa_run_config()
    optimizer = create_gepa_optimizer(
        metric=refinement_gepa_metric,
        config=gepa_run_config,
        reflection_lm=reflection_lm
    )
    
    program_to_optimize = GlmSelfReflect()
    optimized_program = optimizer.compile(
        student=program_to_optimize, 
        trainset=trainset,
        valset=valset,
    )
    
    print(f"Saving optimized program to '{output_file}'")
    optimized_program.save(output_file)
    print("GEPA optimization complete!")

if __name__ == "__main__":
    main()
