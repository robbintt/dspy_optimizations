# GEPA Self-Reflection Optimizer: End-to-End Guide

This guide demonstrates how to use GEPA (Genetic Evolutionary Prompt Algorithm) to optimize a Figure-8 self-reflection architecture. GEPA uses a strong **Reflection LM** to analyze failures and propose textual improvements to your prompts, evolving them to achieve better performance.

## System Overview

The system consists of:
- A **Task LM** that generates answers and critiques
- A **Reflection LM** that GEPA uses to analyze failures and improve prompts
- A self-reflection module (`GlmSelfReflect`) with generator, critic, and refiner components
- GEPA optimizer that evolves the instructions in each component

## Prerequisites

### 1. Install Dependencies

```bash
pip install dspy-ai sentence-transformers pyyaml
```

### 2. Set Up API Keys

Set one or more of the following environment variables based on your model provider:

```bash
export CEREBRAS_API_KEY="your_cerebras_key"
# or
export ZHIPUAI_API_KEY="your_zhipuai_key"
# or
export OPENAI_API_KEY="your_openai_key"
# or
export ANTHROPIC_API_KEY="your_anthropic_key"
```

## Step 1: Configuration

The system uses a YAML configuration file for model settings. Create `config/models.yaml`:

```yaml
# config/models.yaml
# Task model for running the self-reflection loop
task_model:
  provider: cerebras
  name: llama3.1-70b
  temperature: 0.7
  max_tokens: 2048

# Reflection model for GEPA optimization (use a strong model)
reflection_model:
  provider: cerebras
  name: llama3.1-70b
  temperature: 1.0
  max_tokens: 32000

# GEPA optimization profile
gepa_profile: development
```

Available GEPA profiles:
- `development`: Quick testing (80 metric calls)
- `small`: Light optimization (auto="light")
- `medium`: Balanced optimization (auto="medium")
- `large`: Heavy optimization (auto="heavy")

## Step 2: Data Generation

Generate synthetic training data with errors for GEPA to learn from:

```bash
python -m gepa_self_optimizer.generate_data
```

This creates `golden_set.json` with question-answer pairs containing injected errors and corresponding critiques.

## Step 3: Run Optimization

Use the `run_gepa.sh` script to run the optimization. The script提供了灵活的参数选项:

### Basic Usage

```bash
# Use development profile (default)
./gepa_self_optimizer/run_gepa.sh \
  -s gepa_system.GlmSelfReflect \
  -m gepa_config.refinement_gepa_metric \
  -d golden_set.json \
  -o optimized_program.json
```

### Advanced Usage

```bash
# Use medium profile with budget override
./gepa_self_optimizer/run_gepa.sh \
  -p medium \
  -a heavy \
  -s gepa_system.GlmSelfReflect \
  -m gepa_config.refinement_gepa_metric \
  -d golden_set.json \
  -o optimized_program.json

# Use heavy profile for thorough optimization
./gepa_self_optimizer/run_gepa.sh \
  -p large \
  -s gepa_system.GlmSelfReflect \
  -m gepa_config.refinement_gepa_metric \
  -d golden_set.json \
  -o optimized_program.json
```

### Script Parameters

- `-p profile`: GEPA profile (development, small, medium, large)
- `-a auto_override`: Override budget (light, medium, heavy)
- `-s student_module`: Required - DSPy module to optimize
- `-m metric`: Required - Metric function for evaluation
- `-d data_file`: Required - Training data JSON
- `-o output_file`: Required - Save optimized program here

## Step 4: Inspect Results

After optimization completes:
1. The optimized program is saved to your specified output file
2. A summary of evolved instructions is printed to the console
3. You can load and inspect the program:

```python
import json
from gepa_system import GlmSelfReflect

# Load optimized program
program = GlmSelfReflect()
program.load("optimized_program.json")

# Display evolved instructions
print("Critic instructions:", program.critic.signature.instructions)
print("Refiner instructions:", program.refiner.signature.instructions)
```

## Understanding the Optimization

GEPA works by:
1. Running your program on the training set
2. Identifying failed examples with low metric scores
3. Using the Reflection LM to analyze failures and generate feedback
4. Proposing new, improved instructions based on feedback
5. Evaluating new candidates and keeping the best performers
6. Iterating to evolve increasingly effective prompts

The optimized program contains evolved instructions that typically:
- Are more detailed and specific
- Include better guidance for edge cases
- Are tailored to your specific task and metric
- Show improved performance on validation data

## Core System Components

### Configuration (`gepa_config.py`)
- Manages model loading and API keys
- Implements the similarity-based metric
- Provides pre-configured GEPA profiles
- Handles optimizer creation

### System Architecture (`gepa_system.py`)
```python
import dspy
from gepa_config import JUDGE_CONSTITUTION

class Generate(dspy.Signature):
    """Generate a comprehensive answer to a given question, using step-by-step reasoning."""
    question: str = dspy.InputField(desc="The question to be answered.")
    draft_answer: str = dspy.OutputField(desc="A comprehensive, step-by-step answer to the question.")

class ShepherdCritic(dspy.Signature):
    """Act as a ruthless critic. Analyze the draft for errors based on the provided constitution."""
    constitution: str = dspy.InputField(desc="The principles for judging the draft.")
    question: str = dspy.InputField(desc="The question the draft is trying to answer.")
    draft_answer: str = dspy.InputField(desc="The draft answer to be critiqued.")
    critique: str = dspy.OutputField(desc="A list of specific errors found in the draft.")
    severity: str = dspy.OutputField(desc="The severity of the errors: High, Medium, or Low.")

class Refine(dspy.Signature):
    """Rewrite the draft to fix all errors identified in the critique to produce a final, correct answer."""
    question: str = dspy.InputField(desc="The original question.")
    draft_answer: str = dspy.InputField(desc="The draft answer that contains errors.")
    critique: str = dspy.InputField(desc="The list of errors to be fixed.")
    final_answer: str = dspy.OutputField(desc="The refined and correct final answer.")

class GlmSelfReflect(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(Generate)
        self.critic = dspy.Predict(ShepherdCritic)
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
            return dspy.Prediction(answer=draft_answer, critique=critique_pkg.critique, severity=critique_pkg.severity)
```

This clean, modular architecture makes it easy to adapt GEPA optimization to your own tasks and use cases.
