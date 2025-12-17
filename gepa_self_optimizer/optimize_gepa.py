import sys
import dspy
import json
from sentence_transformers import SentenceTransformer, util
from gepa_config import setup_dspy, run_settings
from gepa_system import GlmSelfReflect

import os

# Check if optimized program already exists to prevent rerunning
if os.path.exists("glm_gepa_complete.json"):
    print("✅ Optimized program already exists at 'glm_gepa_complete.json'. Skipping optimization.")
    exit(0)

# --- 1. PROVIDE A WORKING SEMANTIC SIMILARITY FUNCTION ---
# The original 'dspy.evaluate.semantic_similarity' does not exist.
print("🔍 Loading semantic similarity model...")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(text1, text2):
    """Computes cosine similarity between two texts."""
    embeddings = similarity_model.encode([text1, text2], convert_to_tensor=True)
    return util.cos_sim(embeddings[0], embeddings[1]).item()

# --- 2. DEFINE THE METRIC FOR GEPA ---
def refinement_gepa_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    score = semantic_similarity(prediction.answer, example.correct_answer)
    # Return boolean: True if similarity > 0.7 (good enough), False otherwise
    return score > 0.7

# --- 3. CONDITIONALLY LOAD OR RUN OPTIMIZATION ---
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
    gepa_auto_setting = run_settings.get("optimization", {}).get("gepa_auto_setting", "medium")
    optimizer = dspy.GEPA(
        metric=refinement_gepa_metric,
        auto=gepa_auto_setting,
        reflection_lm=reflection_lm,
        track_stats=True
    )
    program_to_optimize = GlmSelfReflect()
    optimized_program = optimizer.compile(
        student=program_to_optimize, 
        trainset=trainset,
        valset=valset,
    )
    
    # --- INSPECT GEPA OPTIMIZATION RESULTS ---
    print("\n🔍 GEPA OPTIMIZATION INSPECTION RESULTS:")
    print("=" * 60)
    
    # Show optimization statistics if available
    if hasattr(optimizer, 'stats') and optimizer.stats:
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
        if hasattr(optimized_program.critic, 'demos') and optimized_program.critic.demos:
            print(f"  Number of demos: {len(optimized_program.critic.demos)}")
            for i, demo in enumerate(optimized_program.critic.demos[:3]):  # Show first 3
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
    
    # --- SAVE RESULTS ---
    optimized_program.save(output_file)
    print(f"\n🏆 GEPA EVOLUTION COMPLETE! Saved to '{output_file}'")

# --- 4. INSPECT RESULTS ---
print("\n--- Inspecting Optimized Program Prompts ---")
try:
    print("--- Optimized Critic Prompts ---")
    print(optimized_program.critic.demos)

    if hasattr(optimized_program, 'refiner'):
        print("\n--- Optimized Refiner Prompts ---")
        print(optimized_program.refiner.demos)
            
except Exception as e:
    print(f"\n[ERROR] Could not inspect prompts due to an unexpected error: {e}")
    print("The program object may be incomplete or structured differently than expected.")
    
sys.stdout.flush()
