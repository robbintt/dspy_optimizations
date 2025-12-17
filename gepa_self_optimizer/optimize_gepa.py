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
        
except AttributeError as e:
    print(f"Could not inspect prompts: {e}")
