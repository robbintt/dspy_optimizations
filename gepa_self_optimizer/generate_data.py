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
        You are a Red Teamer tasked with sabotaging a perfect answer.
        Your goal is to create a flawed version that is DIFFICULT for a top-tier AI to find and fix.

        The flaw must be:
        1.  FATAL: It makes the answer incorrect or unsafe.
        2.  SUBTLE: It should not be an obvious typo or blatant hallucination. It must blend in with the rest of the text.
        3.  PLAUSIBLE: A non-expert might not notice the error.
        4.  HARD-TO-FIX: Fixing the error requires careful reasoning, not just a simple word swap.

        Examples of good flaws:
        -   A subtle but critical logical misstep in a multi-stage argument.
        -   A misapplication of a scientific principle that seems correct on the surface.
        -   A code error that only manifests under specific, non-obvious conditions.
        -   Introducing a convincing but false "alternative fact" that fits the context.

        DO NOT:
        -   Add typos or grammatical mistakes.
        -   Invent obvious nonsense.
        -   Make a simple calculation error.
        -   Change the topic.
        """
        question: str = dspy.InputField(desc="The question the original answer addresses.")
        correct_answer: str = dspy.InputField(desc="The perfect, step-by-step answer to be sabotaged.")
        sabotage_goal: str = dspy.InputField(desc="A specific, challenging instruction from the user to guide the type of sabotage.")
        saboteurs_tactic_log: str = dspy.OutputField(desc="A short, internal note explaining the subtle flaw you decided to inject and why it's hard to spot.")
        bad_draft: str = dspy.OutputField(desc="The rewritten answer containing the subtle, fatal flaw.")
        gold_critique: str = dspy.OutputField(desc="A concise yet precise description of the hidden flaw that the red-teamer created.")

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
                    
                # 3. Dynamically build a sophisticated sabotage goal
                sabotage_goal = feedback_instruction
                
                if not sabotage_goal:
                    # On the first try, provide a challenging but open-ended goal
                    sabotage_goal = (
                        "Your previous error was easily detected. Create a much more subtle and sophisticated flaw. "
                        "Consider a subtle logical fallacy, a misapplied principle, or a plausible but incorrect detail."
                    )
                
                bug_predictor = dspy.ChainOfThought(BugInjector)
                corrupted = bug_predictor(
                    question=base.question,
                    correct_answer=base.correct_answer,
                    sabotage_goal=sabotage_goal
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
                        feedback_instruction = (
                            "THE MODEL FOUND YOUR ERROR. IT WAS NOT SUBTLE ENOUGH. "
                            "You must now create an extremely clever, contextual flaw that is almost invisible, even to an expert. "
                            "Avoid anything that looks like a simple mistake. Think like an adversary trying to poison the model's knowledge."
                        )
                    else: # score < MIN_SCORE
                        print(f"⚫ [Attempt {sabotage_attempt}] Too hard (Score: {score:.2f}). Make the flaw more solvable but still tricky.")
                        feedback_instruction = (
                            "The error you created was too obscure and made the answer nonsensical. "
                            "For this next attempt, create a flaw that is subtle but FAIR, meaning a powerful reasoning model can plausibly find and fix it."
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
