import dspy
from dspy.signatures import Signature
import random
import json
import os
import sys
import time
from gepa_self_optimizer.gepa_config import setup_dspy, refinement_gepa_metric
from .gepa_system import GlmSelfReflect
from dspy.evaluate import Evaluate

#
# --- SEEDING AND RANDOMNESS SETUP ---
# Use the current Unix timestamp in milliseconds to ensure unique runs
# and bust any potential caches that rely on the random seed.
SEED = int(time.time() * 1000)
print(f"🎲 Using millisecond timestamp as seed: {SEED}")

random.seed(SEED)

#
# --- 1. CHECK IF FILE EXISTS FIRST ---
# This avoids expensive API calls if the data is already present.
output_filename = "golden_set.json"
if os.path.exists(output_filename):
    print(f"\n📄 Golden set already exists at {output_filename}. Skipping generation.")
    # Exit early, preventing the rest of the script from running
    sys.exit(0)

# Number of examples to generate
num_examples_to_generate = 25
# --- SIGNATURES ---
class WorldGenerator(Signature):
    """Given a concept, invent a short, self-contained set of unique rules for a fictional system."""
    concept: str = dspy.InputField(desc="A high-level concept for a fictional system, e.g., 'alchemy', 'bureaucracy', 'alien justice'.")
    system_rules: str = dspy.OutputField(desc="A text block describing 3-5 unique, concise, and non-intuitive rules for the system.")
    problem_premise: str = dspy.OutputField(desc="A short premise for a problem to be solved using these rules.")

class ContextualQA(Signature):
    """Based on a provided context and premise, generate a complex question and its perfect answer."""
    system_rules: str = dspy.InputField(desc="The unique rules of the fictional system.")
    problem_premise: str = dspy.InputField(desc="The premise for the question.")
    question: str = dspy.OutputField(desc="A complex multi-step question solvable only by carefully applying the provided rules.")
    correct_answer: str = dspy.OutputField(desc="The perfect, step-by-step answer that correctly applies the system's rules.")

class BugInjector(Signature):
    """
    You are an expert Red Teamer. Your previous attempts at creating a subtle sabotage have FAILED because the system detected them too easily.
    THOROUGHLY ANALYZE the complete history of your failed attempts in 'attempts_history'. Each includes the flawed draft and score.
    
    Based on this analysis, create a fundamentally different and MORE sophisticated sabotage. Learn from each failure:
    - If previous attempts were "too easy", your new flaw must be exponentially more subtle
    - If previous attempts were "too hard", your new flaw must be more discoverable but still challenging
    
    Your new flaw must be:
    1.  FATAL: Makes the answer definitively incorrect
    2.  EXTREMELY SUBTLE: Blended naturally, nearly undetectable
    3.  COGNITIVELY TAXING: Requires deep reasoning to identify and correct
    
    LEARN FROM HISTORY: Study why each previous attempt failed and escalate your sophistication accordingly.
    
    ABSOLUTELY FORBIDDEN:
    -   Reusing error types from failed attempts
    -   Surface-level errors (typos, grammar, obvious nonsense)
    -   Simple calculation mistakes
    -   Anything that would be flagged easily
    """
    question: str = dspy.InputField(desc="The question the original answer addresses.")
    correct_answer: str = dspy.InputField(desc="The perfect, step-by-step answer to be sabotaged.")
    attempts_history: str = dspy.InputField(desc="CRITICAL: Complete history of all failed attempts. Study this deeply to understand what failed and escalate sophistication dramatically. Each entry includes score and why it failed.")
    saboteurs_tactic_log: str = dspy.OutputField(desc="Strategic analysis of your new approach, explaining how you learned from history and why this flaw is fundamentally more sophisticated.")
    bad_draft: str = dspy.OutputField(desc="The rewritten answer containing your most sophisticated, subtle, and hard-to-detect fatal flaw yet.")
    gold_critique: str = dspy.OutputField(desc="A precise description of the new flaw that highlights its advanced subtlety and why it's significantly harder to detect than previous attempts.")

# --- THE FACTORY ---
def generate_synthetic_data(num_examples=25, task_lm=None, reflection_lm=None):
    """
    Generates a dataset using a synthetic world generation feedback loop.
    This avoids reliance on real-world knowledge and increases problem perplexity.
    """
    global SEED
    
    # Create highly unique concepts from combination parts
    adjectives = ["bureaucratic", "sentient", "ancient", "chaotic", "harmonious"]
    subjects = ["alchemy", "dream-crafting", "fungal logic", "interdimensional trade", "time-travel law"]
    contexts = ["galactic empires", "underwater cities", "digital realms", "medieval villages"]
    
    MAX_SCORE = 0.75
    MIN_SCORE = 0.30

    print(f"🧠 Curating {num_examples} examples using novel synthetic contexts...")
    print(f"   Target similarity score range per item: [{MIN_SCORE:.2f}, {MAX_SCORE:.2f})\n")
    
    unoptimized_program = GlmSelfReflect()
    evaluator = Evaluate(devset=[], metric=refinement_gepa_metric, num_threads=1)
    
    good_dataset = []
    total_attempts = 0
    
    while len(good_dataset) < num_examples:
        chosen_concept = f"{random.choice(adjectives)} {random.choice(subjects)} for the context of {random.choice(contexts)}"
        cache_busting_nonce = random.randint(0, 2**63 - 1)
        unique_context = f"concept-{hash(chosen_concept)}-seed-{SEED}-nonce-{cache_busting_nonce}"
        
        try:
            print(f"--- Generating for concept: '{chosen_concept}' ---")

            # 1. GENERATE A SYNTHETIC WORLD
            world_predictor = dspy.ChainOfThought(WorldGenerator)
            world = world_predictor(concept=chosen_concept, unique_id=unique_context)

            # 2. GENERATE A Q&A PAIR BASED ON THE SYNTHETIC WORLD
            qa_predictor = dspy.ChainOfThought(ContextualQA)
            base = qa_predictor(
                system_rules=world.system_rules,
                problem_premise=world.problem_premise
            )
            
            # 2. Enter a feedback loop to find a suitable sabotage for this Q&A
            sabotage_attempt = 0
            item_is_good = False
            # Initialize the history for this new base Q&A pair
            attempts_history = []

            # We will try up to 4 times to get a good error for this one Q&A pair
            while not item_is_good and sabotage_attempt < 4:
                sabotage_attempt += 1
                total_attempts += 1

                bug_predictor = dspy.ChainOfThought(BugInjector)
                # Pass the full history of all previous attempts for this item
                history_string = "\n---\n".join(attempts_history)
                
                try:
                    corrupted = bug_predictor(
                        question=base.question,
                        correct_answer=base.correct_answer,
                        attempts_history=history_string
                    )
                        
                    ex = dspy.Example(
                        question=base.question,
                        draft_answer=corrupted.bad_draft,       
                        gold_critique=corrupted.gold_critique,  
                        correct_answer=base.correct_answer,     
                    ).with_inputs("question", "draft_answer")

                    # 4. Evaluate
                    eval_result = evaluator(unoptimized_program, devset=[ex])
                    score = eval_result.score / 100.0
                        
                    if MIN_SCORE <= score < MAX_SCORE:
                        good_dataset.append(ex)
                        print(f"✅ [{len(good_dataset)}/{num_examples}] KEPT. Score: {score:.2f} (after {sabotage_attempt} tries)")
                        item_is_good = True
                    elif score >= MAX_SCORE:
                        print(f"⚪ [Attempt {sabotage_attempt}] Too easy (Score: {score:.2f}). Trying a more devious error...")
                        history_entry = (
                            f"Attempt {sabotage_attempt} scored {score:.2f}, which was too easy. The system detected the error in this draft: {corrupted.bad_draft}"
                        )
                        attempts_history.append(history_entry)
                    else: # score < MIN_SCORE
                        print(f"⚫ [Attempt {sabotage_attempt}] Too hard (Score: {score:.2f}). Making the flaw more solvable but still tricky.")
                        history_entry = (
                            f"Attempt {sabotage_attempt} scored {score:.2f}, which was too hard. The error in this draft was too obscure: {corrupted.bad_draft}"
                        )
                        attempts_history.append(history_entry)
                
                except Exception as e:
                    print(f"⚠️ [Attempt {sabotage_attempt}] BugInjector prediction failed: {e}. Retrying.")
                    # Add a failure to the history to inform the next BugInjector attempt
                    history_entry = (
                        f"Attempt {sabotage_attempt} failed due to a prediction error: {e}. You must produce a valid output in the next attempt."
                    )
                    attempts_history.append(history_entry)
                    # Continue to the next attempt in the while loop
                    continue
            
            if not item_is_good:
                print(f"❌ Could not find a good error for the concept: '{chosen_concept}' after 4 attempts. Moving on.")

        except Exception as e:
            print(f"❌ A critical error occurred with concept '{chosen_concept}': {e}")
        
    print(f"\n✅ Dataset curated! Found {len(good_dataset)} good examples out of {total_attempts} total attempts.")
    return good_dataset

if __name__ == "__main__":
    num_examples_to_generate = 25

    print("🚀 Initializing DSPy and models to generate golden set...")
    task_lm, reflection_lm = setup_dspy()
    
    # The function now handles generation, curation, and reporting
    with dspy.context(lm=task_lm):
        synthetic_dataset = generate_synthetic_data(num_examples=num_examples_to_generate, task_lm=task_lm, reflection_lm=reflection_lm)

    if synthetic_dataset:
        # Convert dspy.Example objects to plain dictionaries for JSON serialization
        json_ready_data = [dict(example) for example in synthetic_dataset]

        # Save the curated data to the expected file
        with open(output_filename, "w") as f:
            json.dump(json_ready_data, f, indent=4)

        print(f"\n💾 Saved curated dataset to {output_filename}")
        print("\nYou can now run the optimizer: python optimize_gepa.py")
    else:
        print("\n⚠️ No good examples were generated. Please check your configuration.")
