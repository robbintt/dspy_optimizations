import argparse
import json
import logging
import sys
import importlib
from pathlib import Path

import dspy

# Import from our local config and system modules
from gepa_config import setup_dspy, get_gepa_run_config, create_gepa_optimizer
from gepa_system import post_compile_inspection

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def dynamic_import(import_path: str):
    """
    Dynamically imports a class or function from a module path string.
    Example: "gepa_system.GlmSelfReflect" or "gepa_config.refinement_gepa_metric"
    """
    try:
        module_path, item_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, item_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import '{import_path}': {e}")
        sys.exit(1)

def print_optimization_summary(program: dspy.Module):
    """Prints a clean, focused summary of the optimized program's instructions."""
    print("\n" + "=" * 60)
    print("🏆 GEPA OPTIMIZATION SUMMARY")
    print("=" * 60)

    def display_component_instructions(component, component_name):
        print(f"\n📝 Evolved '{component_name}' Component:")
        instruction = ""
        # Handle both dspy.Predict and dspy.ChainOfThought structures
        if hasattr(component, 'predict') and hasattr(component.predict, 'signature'):
            instruction = component.predict.signature.instructions
        elif hasattr(component, 'signature'):
            instruction = component.signature.instructions
        
        if instruction:
            print(f"  Instructions ({len(instruction)} chars):")
            # Limit output to first 200 chars for brevity in console
            print(f"    \"{instruction[:200]}{'...' if len(instruction) > 200 else ''}\"")
        else:
            print("  No instructions found.")

    for attr_name in dir(program):
        if not attr_name.startswith('_'): # Filter private attributes
            attr = getattr(program, attr_name)
            if isinstance(attr, dspy.Module):
                display_component_instructions(attr, attr_name)
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run a GEPA optimization on a DSPy module.")
    
    # --- Configuration ---
    parser.add_argument(
        "--profile", 
        type=str, 
        default="development", 
        help="GEPA profile from gepa_config.py (e.g., 'development', 'medium', 'heavy')."
    )
    parser.add_argument(
        "--override_auto", 
        type=str, 
        choices=["light", "medium", "heavy"], 
        default=None,
        help="Override the budget setting from the profile."
    )
    
    # --- Paths and Imports ---
    parser.add_argument(
        "--student_module", 
        type=str, 
        required=True,
        help="Import path to the dspy.Module to optimize (e.g., 'gepa_system.GlmSelfReflect')."
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        required=True,
        help="Import path to the metric function (e.g., 'gepa_config.refinement_gepa_metric')."
    )
    parser.add_argument(
        "--data_file", 
        type=str, 
        required=True,
        help="Path to the training data JSON file."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True,
        help="Path to save the final optimized program."
    )

    args = parser.parse_args()

    # --- Load Configuration ---
    logger.info(f"Loading GEPA profile: '{args.profile}'")
    gepa_run_config = get_gepa_run_config(args.profile)
    if args.override_auto:
        logger.info(f"Overriding 'auto' budget with: '{args.override_auto}'")
        gepa_run_config.auto = args.override_auto
    
    # --- Setup DSPy and Models ---
    logger.info("Setting up DSPy models...")
    task_lm, reflection_lm = setup_dspy()

    # --- Load Data ---
    logger.info(f"Loading training data from: '{args.data_file}'")
    try:
        with open(args.data_file, "r") as f:
            raw_data = json.load(f)
            trainset = [dspy.Example(**d).with_inputs("question", "draft_answer") for d in raw_data]
            valset = trainset[-5:] 
            trainset = trainset[:-5]
        logger.info(f"Loaded {len(trainset)+len(valset)} examples. Train: {len(trainset)}, Val: {len(valset)}.")
    except FileNotFoundError:
        logger.error(f"Data file not found: {args.data_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load or parse data file: {e}")
        sys.exit(1)

    # --- Dynamic Imports for Student and Metric ---
    StudentClass = dynamic_import(args.student_module)
    metric_fn = dynamic_import(args.metric)

    # --- Run GEPA Optimization ---
    logger.info("Starting GEPA optimization...")
    optimizer = create_gepa_optimizer(
        metric=metric_fn,
        config=gepa_run_config,
        reflection_lm=reflection_lm
    )
    student_program = StudentClass()
    
    optimized_program = optimizer.compile(
        student=student_program, 
        trainset=trainset,
        valset=valset,
    )
    
    # --- Post-Compile Inspection (for debugging) ---
    post_compile_inspection(optimized_program, program_name="GEPA Optimized Program")

    # --- Save and Inspect Results ---
    logger.info(f"Saving optimized program to: '{args.output_file}'")
    try:
        optimized_program.save(args.output_file)
    except Exception as e:
        logger.error(f"Failed to save optimized program: {e}")
        sys.exit(1)

    print_optimization_summary(optimized_program)


if __name__ == "__main__":
    main()
