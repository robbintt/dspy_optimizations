import dspy
import random
import json
from gepa_config import task_lm 

# Use the task model for data generation
with dspy.context(lm=task_lm):
    # --- SIGNATURES ---
    class TopicToQA(dspy.Signature):
        """Generate a complex reasoning question and a perfect step-by-step answer."""
        topic = dspy.InputField()
        question = dspy.OutputField()
        correct_answer = dspy.OutputField()

    class BugInjector(dspy.Signature):
        """Rewrite the answer to include a specific, fatal error. Explain the error."""
        question = dspy.InputField()
        correct_answer = dspy.InputField()
        error_type = dspy.InputField()
        bad_draft = dspy.OutputField(desc="The corrupted answer")
        gold_critique = dspy.OutputField(desc="Precise description of the error")

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
                base = dspy.ChainOfThought(TopicToQA)(topic=topic)
                
                # 2. Inject Bug
                bug = random.choice(sabotage_types)
                corrupted = dspy.ChainOfThought(BugInjector)(
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
