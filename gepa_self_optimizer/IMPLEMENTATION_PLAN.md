### **The End-to-End Guide: 20-Minute GEPA Optimization**

This guide uses GEPA (Genetic Evolutionary Prompt Algorithm) to optimize the prompt and instructions within a Figure-8 self-reflection architecture. GEPA is uniquely powerful because it uses a strong **Reflection LM** to analyze failures and propose textual improvements to your prompt.

#### **Step 0: Terminal Setup**

```bash
# 1. Install libraries
# NOTE: sentence-transformers is required for the semantic similarity function.
pip install dspy-ai sentence-transformers

# 2. Create project folder
mkdir glm_gepa_figure8
cd glm_gepa_figure8
touch config.py system.py generate_data.py optimize_gepa.py
```

#### **File 1: Configuration (`config.py`)**

We define two models: a **Task LM** (for running the self-reflection loop) and a **Reflection LM** (for GEPA to analyze and improve prompts). Using a stronger model for reflection is a GEPA best practice.

```python
# config.py
import os
import yaml
import dspy
from pathlib import Path

# --- 1. LOAD CONFIGURATIONS FROM YAML ---
# Determine the directory of this script to find the config file
CONFIG_DIR = Path(__file__).parent
MODEL_CONFIG_PATH = CONFIG_DIR / "config" / "models.yaml"

# Load the entire model configuration file
with open(MODEL_CONFIG_PATH, "r") as f:
    model_configs = yaml.safe_load(f)

# --- 2. SETUP CEREBRAS API KEY ---
# API key for Cerebras. Set this in your environment.
API_KEY = os.getenv("CEREBRAS_API_KEY", "YOUR_CEREBRAS_API_KEY")
if API_KEY == "YOUR_CEREBRAS_API_KEY":
    raise ValueError("Please set the CEREBRAS_API_KEY environment variable.")


def _create_lm(config_name: str) -> dspy.LM:
    """
    Helper function to create a dspy.LM instance from a named configuration.
    """
    config = model_configs.get(config_name)
    if not config:
        raise ValueError(f"Model configuration '{config_name}' not found in {MODEL_CONFIG_PATH}")
    
    # Construct the model identifier in 'provider/model_name' format
    model_id = f"{config['provider']}/{config['name']}"
    
    # Extract parameters that are not the model identifier itself
    lm_params = {k: v for k, v in config.items() if k not in ['provider', 'name']}
    
    return dspy.LM(model=model_id, api_key=API_KEY, **lm_params)

# --- 3. INSTANTIATE THE LANGUAGE MODELS ---
# The task model for generating and refining
task_lm = _create_lm("task_model")

# The reflection model for GEPA's self-improvement step
reflection_lm = _create_lm("reflection_model")

# Set the default LM for DSPy to our task model
dspy.configure(lm=task_lm)

# --- 2. THE JUDGE'S CONSTITUTION ---
JUDGE_CONSTITUTION = """
You are a Constitutional Critic. Adhere to these principles:
1. FALSEHOODS ARE FATAL: If an answer contains a factual error, mark it INVALID.
2. NO SYCOPHANCY: Do not be polite. Be pedantic.
3. CODE MUST RUN: In code tasks, syntax errors are immediate failures.
4. LOGIC OVER STYLE: Ignore tone; focus on the reasoning chain.
"""
```

#### **File 2: The Data Factory (`generate_data.py`)**

This script creates the synthetic "failures" that GEPA will use to learn how to improve the prompt.

```python
# generate_data.py
import dspy
import random
import json
from config import task_lm 

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
```

#### **File 3: The System (`system.py`)**

This defines the Figure-8 architecture itself. GEPA will mutate the instructions within the signatures of the `critic` and `refiner` modules.

```python
# system.py
import dspy
from config import JUDGE_CONSTITUTION

# --- SIGNATURES ---
class Generate(dspy.Signature):
    """Generate a comprehensive answer. Use System 2 thinking."""
    question = dspy.InputField()
    draft_answer = dspy.OutputField()

class ShepherdCritic(dspy.Signature):
    """Act as a ruthless critic. Analyze the draft for errors based on the Constitution."""
    constitution = dspy.InputField()
    question = dspy.InputField()
    draft_answer = dspy.InputField()
    critique = dspy.OutputField(desc="List of specific errors")
    severity = dspy.OutputField(desc="High, Medium, or Low")

class Refine(dspy.Signature):
    """Rewrite the draft to fix the errors identified in the critique."""
    question = dspy.InputField()
    draft_answer = dspy.InputField()
    critique = dspy.InputField()
    final_answer = dspy.OutputField()

# --- THE MODULE ---
class GlmSelfReflect(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(Generate)
        self.critic = dspy.ChainOfThought(ShepherdCritic)
        self.refiner = dspy.Predict(Refine)

    def forward(self, question, draft_answer=None):
        if not draft_answer:
            draft_answer = self.generator(question=question).draft_answer
        
        critique_pkg = self.critic(
            constitution=JUDGE_CONSTITUTION,
            question=question, 
            draft_answer=draft_answer
        )
        
        if "High" in critique_pkg.severity or "Medium" in critique_pkg.severity:
            final = self.refiner(
                question=question,
                draft_answer=draft_answer,
                critique=critique_pkg.critique
            )
            return dspy.Prediction(answer=final.final_answer, critique=critique_pkg.critique, severity=critique_pkg.severity)
        else:
            # If the critique is low severity, return the original draft.
            return dspy.Prediction(answer=draft_answer, critique=critique_pkg.critique, severity=critique_pkg.severity)
```

#### **File 4: The GEPA Optimizer (`optimize_gepa.py`)**

This is where the magic happens. We run GEPA on the entire program. Unlike the original guide, we optimize the whole system at once, which is more idiomatic for how GEPA works.

```python
# optimize_gepa.py
import dspy
import json
from sentence_transformers import SentenceTransformer, util
from config import task_lm, reflection_lm
from system import GlmSelfReflect

# --- 1. PROVIDE A WORKING SEMANTIC SIMILARITY FUNCTION ---
# The original 'dspy.evaluate.semantic_similarity' does not exist.
print("🔍 Loading semantic similarity model...")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(text1, text2):
    """Computes cosine similarity between two texts."""
    embeddings = similarity_model.encode([text1, text2], convert_to_tensor=True)
    return util.cos_sim(embeddings[0], embeddings[1]).item()

# --- 2. LOAD DATA ---
print("\n📂 Loading Golden Set...")
with open("golden_set.json", "r") as f:
    raw_data = json.load(f)
    trainset = [dspy.Example(**d).with_inputs("question", "draft_answer") for d in raw_data]
    # GEPA also benefits from a small validation set
    valset = trainset[-5:] 
    trainset = trainset[:-5]


# ---------------------------------------------------------
# PHASE: EVOLVE THE ENTIRE SYSTEM WITH GEPA
# ---------------------------------------------------------
print("\n🧬 [SINGLE PHASE] Evolving the GlmSelfReflect system with GEPA...")

# Correct GEPA initialization
# - 'auto' controls the optimization budget (light/medium/heavy).
# - 'reflection_lm' is the strong model used to find weaknesses and propose fixes.
optimizer = dspy.GEPA(
    metric=refinement_gepa_metric,
    auto="medium",
    reflection_lm=reflection_lm, 
    track_stats=True
)

# The program to be optimized
program_to_optimize = GlmSelfReflect()

# Compile and optimize the entire program
# NOTE: We optimize the whole self-reflection loop in one go. This is simpler and
# often more effective with GEPA than optimizing modules in isolation.
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
```

#### **Execution Plan**

1.  **Generate Data:** `python generate_data.py`
2.  **Run Optimization:** `python optimize_gepa.py`

Watch the terminal output. You will see GEPA reflecting on failed traces and using the `reflection_lm` to write new, improved instructions for the `ShepherdCritic` and `Refine` modules. After it completes, `optimize_gepa.py` will print the optimized prompts so you can see exactly how they evolved.
