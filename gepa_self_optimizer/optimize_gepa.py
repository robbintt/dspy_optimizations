import dspy
import json
from sentence_transformers import SentenceTransformer, util
from gepa_config import task_lm, reflection_lm
from gepa_system import GlmSelfReflect

# --- 1. PROVIDE A WORKING SEMANTIC SIMILARITY FUNCTION ---
# The original 'dspy.evaluate.semantic_similarity' does not exist.
print("🔍 Loading semantic similarity model...")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(text1, text2):
    """Computes cosine similarity between two texts."""
    embeddings = similarity_model.encode([text1, text2], convert_to_tensor=True)
    return util.cos_sim(embeddings[0], embeddings[1]).item()

# --- 2. DEFINE THE METRIC FOR GEPA ---
def refinement_gepa_metric(gold, pred, trace=None):
    score = semantic_similarity(pred.answer, gold.correct_answer)
    feedback = f"Similarity score is {score:.2f}. The reference answer is '{gold.correct_answer}'."
    return dspy.evaluate.answer_with_feedback(score, feedback)

# --- 3. LOAD DATA ---
print("\n📂 Loading Golden Set...")
with open("golden_set.json", "r") as f:
    raw_data = json.load(f)
    trainset = [dspy.Example(**d).with_inputs("question", "draft_answer") for d in raw_data]
    valset = trainset[-5:] 
    trainset = trainset[:-5]

# ---------------------------------------------------------
# PHASE: EVOLVE THE ENTIRE SYSTEM WITH GEPA
# ---------------------------------------------------------
print("\n🧬 [SINGLE PHASE] Evolving the GlmSelfReflect system with GEPA...")

optimizer = dspy.GEPA(
    metric=refinement_gepa_metric,
    auto="medium",
    reflection_lm=reflection_lm, 
    track_stats=True
)

program_to_optimize = GlmSelfReflect()

optimized_program = optimizer.compile(
    student=program_to_optimize, 
    trainset=trainset,
    valset=valset,
)

# --- 4. SAVE AND INSPECT RESULTS ---
optimized_program.save("glm_gepa_complete.json")
print("\n🏆 GEPA EVOLUTION COMPLETE! Saved to 'glm_gepa_complete.json'")
print("\n--- What changed? ---")
print("Inspect your optimized program's prompts:")
optimized_program.critic.display()
optimized_program.refiner.display()
