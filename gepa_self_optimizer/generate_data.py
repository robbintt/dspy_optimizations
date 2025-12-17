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
        """Given a topic, generate a complex reasoning question and a perfect, step-by-step answer."""
        topic: str = dspy.InputField(desc="The general topic for the question.")
        question: str = dspy.OutputField(desc="A complex question that requires reasoning.")
        correct_answer: str = dspy.OutputField(desc="The perfect, step-by-step correct answer to the question.")

    class BugInjector(Signature):
        """
        You are a Red Teamer. Your last attempt at creating a subtle sabotage FAILED because the system found it too easily.
        Analyze the failed attempt provided in 'last_failed_attempt' and create a NEW, much more sophisticated sabotage.
        Your new flaw must be:
        1.  FATAL: It makes the answer incorrect.
        2.  SUBTLE: It must not be obvious. Blend it in.
        3.  HARD-TO-FIX: Require careful reasoning to spot and correct.
        
        DO NOT:
        -   Add typos or grammatical mistakes.
        -   Invent obvious nonsense.
        -   Make simple calculation errors.
        """
        question: str = dspy.InputField(desc="The question the original answer addresses.")
        correct_answer: str = dspy.InputField(desc="The perfect, step-by-step answer to be sabotaged.")
        last_failed_attempt: str = dspy.InputField(desc="The previous failed sabotage attempt, including the critique of why it was too easy.")
        saboteurs_tactic_log: str = dspy.OutputField(desc="A short, internal note explaining the NEW subtle flaw and why it's harder to spot.")
        bad_draft: str = dspy.OutputField(desc="The rewritten answer containing the NEW, more subtle, fatal flaw.")
        gold_critique: str = dspy.OutputField(desc="A concise description of the NEW, harder-to-find flaw.")

    # --- THE FACTORY ---
    def generate_synthetic_data(num_examples=25):
        """
        Generates and curates a dataset using a feedback loop. If an example is
        too easy or too hard, the script provides this feedback to the model
        in the next attempt to guide it toward a better result.
        """
        topics = ["Python Recursion", "Thermodynamics", "SQL Joins", "Bayesian Stats", "Game Theory", "Roman History"]
        
        MAX_SCORE = 0.75
        MIN_SCORE = 0.30

        print(f"🧠 Curating {num_examples} examples with adaptive feedback...")
        print(f"   Target score range per item: [{MIN_SCORE:.2f}, {MAX_SCORE:.2f})\n")
        
        setup_dspy()
        unoptimized_program = GlmSelfReflect()
        evaluator = Evaluate(devset=[], metric=refinement_gepa_metric, num_threads=1)
        
        good_dataset = []
        total_attempts = 0

        while len(good_dataset) < num_examples:
            topic_idx = len(good_dataset) # Use count to cycle through topics
            topic = topics[topic_idx % len(topics)]
            
            try:
                # 1. Generate a single, high-quality base Q&A pair
                base_predictor = dspy.ChainOfThought(TopicToQA)
                base = base_predictor(topic=topic)
                
                # 2. Enter a feedback loop to find a suitable sabotage for this Q&A
                sabotage_attempt = 0
                feedback_instruction = ""
                item_is_good = False

                # We will try up to 4 times to get a good error for this one Q&A pair
                while not item_is_good and sabotage_attempt < 4:
                    total_attempts += 1
                    sabotage_attempt += 1
                    
            last_failure_report = "" # Initialize for the first run
                
            bug_predictor = dspy.ChainOfThought(BugInjector)
            corrupted = bug_predictor(
                question=base.question,
                correct_answer=base.correct_answer,
                last_failed_attempt=last_failure_report
            )
                    
                ex = dspy.Example(
                    question=base.question,
                    draft_answer=corrupted.bad_draft,       
                    gold_critique=corrupted.gold_critique,  
                    correct_answer=base.correct_answer,     
                ).with_inputs("question", "draft_answer")

                # 4. Evaluate and provide feedback for the next loop
                eval_result = evaluator(unoptimized_program, devset=[ex])
                score = eval_result.score / 100.0
                    
                if MIN_SCORE <= score < MAX_SCORE:
                    good_dataset.append(ex)
                    print(f"✅ [{len(good_dataset)}/{num_examples}] KEPT. Score: {score:.2f} (after {sabotage_attempt} tries)")
                    item_is_good = True
                elif score >= MAX_SCORE:
                    print(f"⚪ [Attempt {sabotage_attempt}] Too easy (Score: {score:.2f}). Demanding a much harder, more devious error...")
                    # Build a report of the failure to give back to the model
                    last_failure_report = (
                        f"YOUR LAST ATTEMPT FAILED. It was scored {score:.2f} and deemed 'too easy'.\n"
                        f"--- YOUR LAST FLAWED DRAFT ---\n{corrupted.bad_draft}\n--- END DRAFT ---\n"
                        f"REASON FOR FAILURE: The system easily detected and corrected your error. "
                        f"You must create something far more subtle that a top-tier AI will miss."
                    )
                else: # score < MIN_SCORE
                    print(f"⚫ [Attempt {sabotage_attempt}] Too hard (Score: {score:.2f}). Make the flaw more solvable but still tricky.")
                    last_failure_report = (
                        f"YOUR LAST ATTEMPT FAILED. It was scored {score:.2f} and deemed 'too hard'.\n"
                        f"--- YOUR LAST FLAWED DRAFT ---\n{corrupted.bad_draft}\n--- END DRAFT ---\n"
                        f"REASON FOR FAILURE: The error was too obscure or nonsensical. "
                        f"You must create a flaw that is subtle but FAIR, meaning a powerful model can plausibly find and fix it."
                    )

                if not item_is_good:
                    print(f"❌ Could not find a good error for the topic: '{topic}' after 4 attempts. Moving on.")

            except Exception as e:
                print(f"❌ A critical error occurred with topic '{topic}': {e}")

        print(f"\n✅ Dataset curated! Found {len(good_dataset)} good examples out of {total_attempts} total feedback-driven attempts.")
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
