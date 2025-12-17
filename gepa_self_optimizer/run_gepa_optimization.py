import argparse
import json
import sys
import importlib
import os
from datetime import datetime

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


def create_run_directory():
    """Creates a timestamped directory for storing run results."""
    # Create a top-level data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create a unique, timestamped subdirectory for this specific run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Add process ID for uniqueness when running multiple scripts concurrently
    pid = os.getpid()
    run_dir = os.path.join(data_dir, f"gepa_run_{timestamp}_{pid}")
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir

def load_data(data_file: str):
    with open(data_file, "r") as f:
        raw_data = json.load(f)
        trainset = [dspy.Example(**d).with_inputs("question", "draft_answer") for d in raw_data]
        valset = trainset[-5:] 
        trainset = trainset[:-5]
    return trainset, valset

def save_run_results(program, valset, metric_fn, run_dir):
    """Saves detailed stats and evaluation metrics to the run directory."""
    if not hasattr(program, 'detailed_results'):
        print("\n📊 GEPA stats were not tracked for this run.")
        return

    print(f"\n📊 Saving detailed run results to '{run_dir}'...")
    
    # 1. Save the detailed GEPA optimization statistics
    detailed_results_path = os.path.join(run_dir, "gepa_detailed_results.json")
    
    # Build the dictionary manually to handle DSPy module objects in candidates
    results_dict = {
        "parents": program.detailed_results.parents,
        "val_aggregate_scores": program.detailed_results.val_aggregate_scores,
        "val_subscores": program.detailed_results.val_subscores,
        "per_val_instance_best_candidates": [list(s) for s in program.detailed_results.per_val_instance_best_candidates],
        "discovery_eval_counts": program.detailed_results.discovery_eval_counts,
        "best_outputs_valset": program.detailed_results.best_outputs_valset,
        "total_metric_calls": program.detailed_results.total_metric_calls,
        "num_full_val_evals": program.detailed_results.num_full_val_evals,
        "log_dir": program.detailed_results.log_dir,
        "seed": program.detailed_results.seed,
    }
    
    # To save the candidate programs, we must convert the module objects to their dict representation
    candidates_as_dicts = []
    for candidate_module in program.detailed_results.candidates:
        # Get the instructions for each predictor within the candidate module
        candidate_dict = {}
        for name, predictor in candidate_module.named_predictors():
            if hasattr(predictor, 'signature'):
                candidate_dict[name] = predictor.signature.instructions
        candidates_as_dicts.append(candidate_dict)

    results_dict["candidates"] = candidates_as_dicts
    results_dict["best_idx"] = program.detailed_results.best_idx
    
    with open(detailed_results_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"  - Saved detailed GEPA stats to '{detailed_results_path}'")

    # 2. Save the final evaluation metrics on the validation set
    if valset and metric_fn:
        print("  - Calculating final metrics on the validation set...")
        from dspy.evaluate import Evaluate
        
        # The detailed_results also contains the best outputs on the valset if track_best_outputs was True
        if program.detailed_results.best_outputs_valset:
            print("  - Using best outputs tracked during GEPA optimization.")
            metric_outputs = []
            # The structure is a list of (candidate_idx, [dspy.Prediction]) for each val instance
            # We need to find the best candidate's outputs
            best_idx = program.detailed_results.best_idx
            best_predictions = []
            for val_task_outputs in program.detailed_results.best_outputs_valset:
                for cand_idx, predictions in val_task_outputs:
                    if cand_idx == best_idx and predictions:
                        best_predictions.append(predictions[0])
                        break
        else:
            print("  - Re-running evaluation with the optimized program.")
            evaluator = Evaluate(devset=valset, metric=metric_fn, num_threads=1, display_progress=False)
            best_predictions = [program(**ex.without('correct_answer', 'gold_critique').inputs()) for ex in valset]


        # Re-calculate the score for logging
        final_score = 0.0
        if best_predictions:
            scores = [metric_fn(example, pred).score for example, pred in zip(valset, best_predictions)]
            final_score = sum(scores) / len(scores)
        
        metrics_to_save = {
            "final_val_score": final_score,
            "val_set_len": len(valset),
            "best_gepa_candidate_idx": program.detailed_results.best_idx,
            "per_instance_scores": [metric_fn(example, pred).score for example, pred in zip(valset, best_predictions)] if best_predictions else []
        }

        metrics_path = os.path.join(run_dir, "validation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_to_save, f, indent=4)
        print(f"  - Saved validation metrics to '{metrics_path}'")


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

    # 1. Create a directory for this specific run's data
    run_dir = create_run_directory()
    print(f"--- GEPA Run Configuration ---")
    print(f"Run Directory:      {run_dir}")

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
    
    # 2. Save the primary optimized program artifact
    optimized_program.save(args.output_file)
    print(f"\n✅ Saved optimized program to '{args.output_file}'")
    
    # 3. Save detailed statistics and metrics for this run
    save_run_results(optimized_program, valset, metric_fn, run_dir)
    
    print_optimization_summary(optimized_program)


if __name__ == "__main__":
    main()
