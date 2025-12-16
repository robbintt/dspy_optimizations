import dspy
from dspy.signatures import Signature
import random
import json
from gepa_config import setup_dspy, task_lm

# Initialize DSPy and models first
setup_dspy()

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
        topics = ["Python Recursion", "Thermodynamics", "SQL Joins", "Bayesian Stats", "Game Theory", "Roman History"] * 5
        dataset = []
        
        print(f"🏭 Manufacturing {num_examples} sabotage examples...")
        
        sabotage_types = ["Math Calculation Error", "Logical Fallacy", "Factual Hallucination", "Code Syntax Error"]

        for i in range(num_examples):
            topic = topics[i % len(topics)]
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
                
                # 3. Package as a dspy.Example with the required .with_inputs()
                ex = dspy.Example(
                    question=base.question,
                    draft_answer=corrupted.bad_draft,       
                    gold_critique=corrupted.gold_critique,  
                    correct_answer=base.correct_answer,     
                ).with_inputs("question", "draft_answer")
                
                dataset.append(ex)
                print(f"✅ [{i+1}/{num_examples}] Sabotaged '{topic}' with {bug}")
                
            except Exception as e:
                print(f"❌ Failed on {topic}: {e}")

        return dataset

if __name__ == "__main__":
    data = generate_synthetic_data(25) 
    serialized = [x.toDict() for x in data]
    with open("golden_set.json", "w") as f:
        json.dump(serialized, f, indent=2)
    print("💾 Saved to golden_set.json")
