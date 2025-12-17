import sys
import dspy
import json
from gepa_config import setup_dspy, refinement_gepa_metric, get_default_gepa_run_config, create_gepa_optimizer
from gepa_system import GlmSelfReflect

import os

# --- 1. PROVIDE A WORKING SEMANTIC SIMILARITY FUNCTION ---
# The original 'dspy.evaluate.semantic_similarity' does not exist.
print("🔍 Loading semantic similarity model...")

# --- 2. CONDITIONALLY LOAD OR RUN OPTIMIZATION ---
output_file = "glm_gepa_complete.json"

if os.path.exists(output_file):
    print(f"✅ Optimized program found at '{output_file}'. Skipping optimization and loading from file.")
    optimized_program = GlmSelfReflect()
    optimized_program.load(output_file)
else:
    print("\n📂 No pre-optimized program found. Starting new optimization run...")
    
    # --- LOAD DATA ---
    print("\n📂 Loading Golden Set...")
    with open("golden_set.json", "r") as f:
        raw_data = json.load(f)
        trainset = [dspy.Example(**d).with_inputs("question", "draft_answer") for d in raw_data]
        valset = trainset[-5:] 
        trainset = trainset[:-5]

    # --- EVOLVE THE ENTIRE SYSTEM WITH GEPA ---
    print("\n🧬 [SINGLE PHASE] Evolving the GlmSelfReflect system with GEPA...")
    task_lm, reflection_lm = setup_dspy()

    # Load the GEPA configuration based on the 'gepa_profile' in settings.yaml
    gepa_run_config = get_default_gepa_run_config()
    print(f"  -> GEPA Profile: '{gepa_run_config.gepa_profile or 'custom'}'")
    if gepa_run_config.auto:
        print(f"  -> GEPA Budget: '{gepa_run_config.auto}'")
    else:
        print(f"  -> GEPA Max Metric Calls: {gepa_run_config.max_metric_calls}")

    # Create the GEPA optimizer using the loaded configuration
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
    
    # --- ADD THIS DEBUGGING BLOCK ---
    print("\n🐛 [DEBUG] Inspecting demos in memory BEFORE saving...")
    critic_demos_count = len(optimized_program.critic.predict.demos) if hasattr(optimized_program.critic, 'predict') and hasattr(optimized_program.critic.predict, 'demos') else 'N/A'
    print(f"  Critic demos found: {critic_demos_count}")
    refiner_demos_count = len(optimized_program.refiner.demos) if hasattr(optimized_program.refiner, 'demos') else 'N/A'
    print(f"  Refiner demos found: {refiner_demos_count}")
    # --- END DEBUGGING BLOCK ---
    
    # --- SAVE RESULTS ---
    optimized_program.save(output_file)
    print(f"\n🏆 GEPA EVOLUTION COMPLETE! Saved to '{output_file}'")

# --- 4. INSPECT GEPA OPTIMIZATION RESULTS ---
print("\n🔍 GEPA OPTIMIZATION INSPECTION RESULTS:")
print("=" * 60)

# Show optimization statistics if available
if 'optimizer' in locals() and hasattr(optimizer, 'stats') and optimizer.stats:
    print("📊 Optimization Statistics:")
    for key, value in optimizer.stats.items():
        print(f"  {key}: {value}")
    print()

# Show the evolved program structure
print("🏗️ Evolved Program Structure:")
print(f"  Program type: {type(optimized_program).__name__}")

# Inspect critic component
if hasattr(optimized_program, 'critic'):
    print("\n📝 Evolved Critic Component:")
    critic_demos = None
    if hasattr(optimized_program.critic, 'predict') and hasattr(optimized_program.critic.predict, 'demos'):
        critic_demos = optimized_program.critic.predict.demos

    if critic_demos:
        print(f"  Number of demos: {len(critic_demos)}")
        for i, demo in enumerate(critic_demos[:3]):  # Show first 3
            print(f"  Demo {i+1}:")
            for key, value in demo.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"    {key}: {value[:100]}...")
                else:
                    print(f"    {key}: {value}")
    else:
        print("  No demos found in critic")

# Inspect refiner component if it exists
if hasattr(optimized_program, 'refiner'):
    print("\n🔧 Evolved Refiner Component:")
    if hasattr(optimized_program.refiner, 'demos') and optimized_program.refiner.demos:
        print(f"  Number of demos: {len(optimized_program.refiner.demos)}")
        for i, demo in enumerate(optimized_program.refiner.demos[:3]):  # Show first 3
            print(f"  Demo {i+1}:")
            for key, value in demo.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"    {key}: {value[:100]}...")
                else:
                    print(f"    {key}: {value}")
    else:
        print("  No demos found in refiner")

# Show any other components
for attr_name in dir(optimized_program):
    if not attr_name.startswith('_') and attr_name not in ['critic', 'refiner', 'save', 'load', 'forward']:
        attr = getattr(optimized_program, attr_name)
        if hasattr(attr, 'demos'):
            print(f"\n📋 Component '{attr_name}':")
            if attr.demos:
                print(f"  Number of demos: {len(attr.demos)}")
            else:
                print("  No demos found")

print("=" * 60)

# --- 5. INSPECT RESULTS ---
print("\n--- Inspecting Optimized Program Prompts ---")
try:
    print("--- Optimized Critic Prompts ---")
    if hasattr(optimized_program.critic, 'predict') and hasattr(optimized_program.critic.predict, 'demos'):
        print(optimized_program.critic.predict.demos)
    else:
        print("No critic demos found")

    if hasattr(optimized_program, 'refiner'):
        print("\n--- Optimized Refiner Prompts ---")
        print(optimized_program.refiner.demos)
            
except Exception as e:
    print(f"\n[ERROR] Could not inspect prompts due to an unexpected error: {e}")
    print("The program object may be incomplete or structured differently than expected.")
    
sys.stdout.flush()
