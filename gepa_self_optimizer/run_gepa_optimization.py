import argparse
import json
import sys
import importlib

import dspy

from gepa_config import setup_dspy, get_gepa_run_config, create_gepa_optimizer
from gepa_system import post_compile_inspection

def dynamic_import(import_path: str):
    module_path, item_name = import_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, item_name)

def print_optimization_summary(program: dspy.Module):
    print("\n" + "=" * 60)
    print("🏆 GEPA OPTIMIZATION SUMMARY")
    print("=" * 60)

    for attr_name in dir(program):
        if not attr_name.startswith('_'):
            attr = getattr(program, attr_name)
            if isinstance(attr, dspy.Module):
                instruction = ""
                if hasattr(attr, 'predict') and hasattr(attr.predict, 'signature'):
                    instruction = attr.predict.signature.instructions
                elif hasattr(attr, 'signature'):
                    instruction = attr.signature.instructions
                
                if instruction:
                    print(f"\n📝 Evolved '{attr_name}' Component:")
                    print(f"  Instructions ({len(instruction)} chars):")
                    print(f"    \"{instruction[:200]}{'...' if len(instruction) > 200 else ''}\"")
    
    print("=" * 60)


def load_data(data_file: str):
    with open(data_file, "r") as f:
        raw_data = json.load(f)
        trainset = [dspy.Example(**d).with_inputs("question", "draft_answer") for d in raw_data]
        valset = trainset[-5:] 
        trainset = trainset[:-5]
    return trainset, valset

def main():
    parser = argparse.ArgumentParser(description="Run a GEPA optimization on a DSPy module.")
    parser.add_argument("--profile", type=str, default="development", 
                       help="GEPA profile from gepa_config.py")
    parser.add_argument("--override_auto", type=str, choices=["light", "medium", "heavy"], 
                       default=None, help="Override the budget setting")
    parser.add_argument("--student_module", type=str, required=True,
                       help="Import path to the dspy.Module to optimize")
    parser.add_argument("--metric", type=str, required=True,
                       help="Import path to the metric function")
    parser.add_argument("--data_file", type=str, required=True,
                       help="Path to the training data JSON file")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to save the optimized program")

    args = parser.parse_args()

    gepa_run_config = get_gepa_run_config(args.profile)
    if args.override_auto:
        gepa_run_config.auto = args.override_auto
    
    task_lm, reflection_lm = setup_dspy()
    trainset, valset = load_data(args.data_file)
    
    StudentClass = dynamic_import(args.student_module)
    metric_fn = dynamic_import(args.metric)

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
    
    post_compile_inspection(optimized_program, program_name="GEPA Optimized Program")
    optimized_program.save(args.output_file)
    print_optimization_summary(optimized_program)


if __name__ == "__main__":
    main()
