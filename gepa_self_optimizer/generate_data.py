import dspy
from dspy.signatures import Signature
import random
import json
import os
import sys
import time
from gepa_config import setup_dspy, task_lm, refinement_gepa_metric
from gepa_system import GlmSelfReflect
from dspy.evaluate import Evaluate

# --- 1. CHECK IF FILE EXISTS FIRST ---
# This avoids expensive API calls if the data is already present.
output_filename = "golden_set.json"
if os.path.exists(output_filename):
    print(f"\n📄 Golden set already exists at {output_filename}. Skipping generation.")
    # Exit early, preventing the rest of the script from running
    sys.exit(0)

# --- 2. INITIALIZE AND GENERATE DATA ---
# Only run this if the file was not found.
print("🚀 Initializing DSPy and models to generate golden set...")
setup_dspy()

# Number of examples to generate
num_examples_to_generate = 25

# Use the task model for data generation
with dspy.context(lm=task_lm):
    # --- SIGNATURES ---
    class TopicToQA(Signature):
        """Given a topic, generate an extremely complex, multi-step reasoning question that contains subtle pitfalls or common misconceptions, and provide a perfect, step-by-step answer. The question should require deep analytical thinking and careful attention to detail."""
        topic: str = dspy.InputField(desc="The general topic for the question.")
        unique_id: str = dspy.InputField(desc="A unique identifier to ensure a novel response is generated.")
        question: str = dspy.OutputField(desc="An extremely complex and multi-step question with subtle pitfalls that requires deep analytical reasoning and careful attention to detail.")
        correct_answer: str = dspy.OutputField(desc="The perfect, step-by-step correct answer to the complex question.")

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
    def generate_synthetic_data(num_examples=25):
        """
        Generates and curates a dataset using a feedback loop. If an example is
        too easy or too hard, the script provides the full history of attempts to the model
        to guide it toward a better result.
        """
        topics = ["Python Recursion", "Thermodynamics", "SQL Joins", "Bayesian Stats", "Game Theory", "Roman History"]
        
        MAX_SCORE = 0.75
        MIN_SCORE = 0.30

        print(f"🧠 Curating {num_examples} examples with full historical feedback...")
        print(f"   Target score range per item: [{MIN_SCORE:.2f}, {MAX_SCORE:.2f})\n")
        
        setup_dspy()
        unoptimized_program = GlmSelfReflect()
        evaluator = Evaluate(devset=[], metric=refinement_gepa_metric, num_threads=1)
        
        good_dataset = []
        total_attempts = 0
        overall_topic_attempts = 0
        
        while len(good_dataset) < num_examples:
            topic_idx = overall_topic_attempts % len(topics)
            topic = topics[topic_idx]
            overall_topic_attempts += 1
            
            try:
                # 1. Generate a single, high-quality base Q&A pair
                base_predictor = dspy.ChainOfThought(TopicToQA)
                base = base_predictor(topic=topic, unique_id=str(overall_topic_attempts))
                
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
                    
                    # Append the result of this attempt to the history for the next loop
                    history_entry = (
                        f"Attempt {sabotage_attempt} scored {score:.2f}, which was too easy. "
                        f"The system detected the error in this draft: {corrupted.bad_draft}"
                    )
                    attempts_history.append(history_entry)
                        
                    if MIN_SCORE <= score < MAX_SCORE:
                        good_dataset.append(ex)
                        print(f"✅ [{len(good_dataset)}/{num_examples}] KEPT. Score: {score:.2f} (after {sabotage_attempt} tries)")
                        item_is_good = True
                    elif score >= MAX_SCORE or score > 0.99:
                        print(f"⚪ [Attempt {sabotage_attempt}] Too easy (Score: {score:.2f}). Trying a more devious error...")
                    else: # score < MIN_SCORE
                        print(f"⚫ [Attempt {sabotage_attempt}] Too hard (Score: {score:.2f}). Making the flaw more solvable but still tricky.")
                        # Update the history entry to reflect 'too hard'
                        attempts_history[-1] = (
                            f"Attempt {sabotage_attempt} scored {score:.2f}, which was too hard. "
                            f"The error in this draft was too obscure: {corrupted.bad_draft}"
                        )
                
                if not item_is_good:
                    print(f"❌ Could not find a good error for the topic: '{topic}' after 4 attempts. Moving on.")

            except Exception as e:
                print(f"❌ A critical error occurred with topic '{topic}': {e}")
            
        print(f"\n✅ Dataset curated! Found {len(good_dataset)} good examples out of {total_attempts} total attempts.")
        return good_dataset

if __name__ == "__main__":
    num_examples_to_generate = 25
    
    # The function now handles generation, curation, and reporting
    synthetic_dataset = generate_synthetic_data(num_examples=num_examples_to_generate)

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
