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

# Load run settings
run_settings = _load_run_settings()

# Use the task model for data generation
with dspy.context(lm=task_lm):
    # --- SIGNATURES ---
    class TopicToQA(Signature):
        """Given a topic, generate a complex reasoning question and a perfect, step-by-step answer."""
        topic: str = dspy.InputField(desc="The general topic for the question.")
        question: str = dspy.OutputField(desc="A complex question that requires reasoning.")
        correct_answer: str = dspy.OutputField(desc="The perfect, step-by-step correct answer to the question.")

    class BugInjector(Signature):
        """Given a correct answer, rewrite it to include a specific, fatal error and explain that error."""
        question: str = dspy.InputField(desc="The question associated with the answer.")
        correct_answer: str = dspy.InputField(desc="The correct answer to be sabotaged.")
        error_type: str = dspy.InputField(desc="The type of error to inject (e.g., 'Math Calculation Error').")
        bad_draft: str = dspy.OutputField(desc="The rewritten answer containing a fatal error.")
        gold_critique: str = dspy.OutputField(desc="A precise description of the injected error.")

    # --- THE FACTORY ---
    def generate_synthetic_data(num_examples=25):
        """
        Generates and curates a dataset item-by-item, keeping only examples
        that fall within a target difficulty score range.
        """
        topics = ["Python Recursion", "Thermodynamics", "SQL Joins", "Bayesian Stats", "Game Theory", "Roman History"]
        sabotage_types = ["Math Calculation Error", "Logical Fallacy", "Factual Hallucination", "Code Syntax Error"]
        
        # --- Score thresholds for a "good" training example ---
        # - MAX_SCORE: If the model scores higher, the example is too easy and provides no learning signal.
        # - MIN_SCORE: If the model scores lower, the example may be too hard or ill-posed for this model.
        MAX_SCORE = 0.75
        MIN_SCORE = 0.30

        print(f"🏭 Curating {num_examples} high-quality examples (this will take longer but yields better data)...")
        print(f"   Target score range per item: [{MIN_SCORE:.2f}, {MAX_SCORE:.2f})\n")
        
        # --- SETUP SYSTEM FOR VALIDATION ---
        # We need the model and system to be available for a quick test of each item
        setup_dspy()
        unoptimized_program = GlmSelfReflect()
        
        # We run the evaluator on one item at a time
        evaluator = Evaluate(
            devset=[],  # Devset will be provided per call
            metric=refinement_gepa_metric,
            num_threads=1,
            display_progress=False, # Keep output clean
            display_table=False,
        )
        
        good_dataset = []
        attempts = 0
        
        while len(good_dataset) < num_examples:
            attempts += 1
            topic = topics[attempts % len(topics)]
            
            try:
                # 1. Generate Truth
                base_predictor = dspy.ChainOfThought(TopicToQA)
                base = base_predictor(topic=topic)
                
                # 2. Inject Bug
                bug = random.choice(sabotage_types)
                bug_predictor = dspy.ChainOfThought(BugInjector)
                corrupted = bug_predictor(
                    question=base.question,
                    correct_answer=base.correct_answer,
                    error_type=bug
                )
                
                # 3. Package as a dspy.Example
                ex = dspy.Example(
                    question=base.question,
                    draft_answer=corrupted.bad_draft,       
                    gold_critique=corrupted.gold_critique,  
                    correct_answer=base.correct_answer,     
                ).with_inputs("question", "draft_answer")

                # 4. IMMEDIATELY VALIDATE THE ITEM
                # The evaluator returns a score as a percentage (0-100)
                eval_result = evaluator(unoptimized_program, devset=[ex])
                score = eval_result.score / 100.0
                
                # 5. JUDGE AND KEEP/DISCARD BASED ON SCORE
                if MIN_SCORE <= score < MAX_SCORE:
                    good_dataset.append(ex)
                    print(f"✅ [{len(good_dataset)}/{num_examples}] KEPT.   Score: {score:.2f}")
                elif score >= MAX_SCORE:
                    print(f"⚪ [{len(good_dataset)}/{num_examples}] DISCARDED (too easy). Score: {score:.2f}")
                else: # score < MIN_SCORE
                    print(f"⚫ [{len(good_dataset)}/{num_examples}] DISCARDED (too hard). Score: {score:.2f}")
                    
            except Exception as e:
                print(f"❌ Generation failed on attempt {attempts}: {e}")

        print(f"\n✅ Dataset curated! Found {num_examples} good examples out of {attempts} total attempts.")
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
