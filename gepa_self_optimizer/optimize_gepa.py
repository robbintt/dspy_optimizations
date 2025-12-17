import sys
import dspy
import json
from gepa_config import setup_dspy, refinement_gepa_metric, get_default_gepa_run_config, create_gepa_optimizer
from gepa_system import GlmSelfReflect, post_compile_inspection

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
        # --- INSTRUMENTATION ---
        print(f"\n[DATA CHECK] Training set size: {len(trainset)}")
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
    
    # --- WORKAROUND: Resolve potential GEPA state desynchronization ---
    print("\n🔧 [WORKAROUND] Re-saving and reloading the optimized program to fix potential state desync...")
    temp_file = "temp_optimized_program.json"
    optimized_program.save(temp_file)
    
    print("   -> Creating new program instance and loading from saved state...")
    cleaned_program = GlmSelfReflect()
    cleaned_program.load(temp_file)
    os.remove(temp_file) # Clean up the temporary file
    print("   -> Workaround complete. Using the cleaned program for inspection.")
    optimized_program = cleaned_program # Replace the old program with the cleaned one
    
    # --- POST-COMPILE DEBUGGING ---
    # Inspect the state of the program immediately after compilation
    post_compile_inspection(optimized_program, program_name="GEPA Optimized & Cleaned Program")
    
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

# Helper to display instructions
def display_component_instructions(component, component_name):
    print(f"\n📝 Evolved {component_name} Component:")
    instruction = ""
    # Handle both dspy.Predict and dspy.ChainOfThought structures
    if hasattr(component, 'predict') and hasattr(component.predict, 'signature'):
        instruction = component.predict.signature.instructions
    elif hasattr(component, 'signature'):
        instruction = component.signature.instructions
    
    if instruction:
        print(f"  Instructions ({len(instruction)} chars): {instruction[:200]}...")
    else:
        print("  No instructions found.")

# Inspect critic component
if hasattr(optimized_program, 'critic'):
    display_component_instructions(optimized_program.critic, "Critic")

# Inspect refiner component if it exists
if hasattr(optimized_program, 'refiner'):
    display_component_instructions(optimized_program.refiner, "Refiner")

# Show any other components
for attr_name in dir(optimized_program):
    if not attr_name.startswith('_') and attr_name not in ['critic', 'refiner', 'save', 'load', 'forward', 'generator']:
        attr = getattr(optimized_program, attr_name)
        if hasattr(attr, 'signature'):
            display_component_instructions(attr, attr_name.title())

print("=" * 60)

# --- 5. INSPECT RESULTS ---
print("\n--- Inspecting Evolved Instructions ---")
try:
    print("\n--- Evolved Critic Instructions ---")
    critic_instruction = ""
    if hasattr(optimized_program.critic, 'predict') and hasattr(optimized_program.critic.predict, 'signature'):
        critic_instruction = optimized_program.critic.predict.signature.instructions
    print(critic_instruction if critic_instruction else "No Critic instructions found.")

    if hasattr(optimized_program, 'refiner'):
        print("\n--- Evolved Refiner Instructions ---")
        refiner_instruction = ""
        if hasattr(optimized_program.refiner, 'signature'):
            refiner_instruction = optimized_program.refiner.signature.instructions
        print(refiner_instruction if refiner_instruction else "No Refiner instructions found.")
            
except Exception as e:
    print(f"\n[ERROR] Could not inspect instructions due to an unexpected error: {e}")
    print("The program object may be incomplete or structured differently than expected.")
    
sys.stdout.flush()
