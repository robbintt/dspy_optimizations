# GEPA Self-Optimizer

A hands-on example of using DSPy's **GEPA (Genetic Evolutionary Prompt Algorithm)** to automatically evolve and optimize the prompts of a Figure-8 self-reflection AI system.

This project demonstrates a powerful prompt engineering technique where a stronger "Reflection" model analyzes failures and proposes textual improvements to a "Task" model's prompts, creating a system that learns to be better over time.

---

## ✨ Features

-   **🧬 Genetic Evolutionary Prompt Algorithm (GEPA):** Leverages DSPy's built-in GEPA optimizer for automated prompt discovery and improvement.
-   **🔄 Figure-8 Self-Reflection Architecture:** Implements a cycle of generation, critique, and refinement to improve answer quality.
-   **🚀 Cerebras Integration:** Optimized to use the Cerebras API with the `zai-glm-4.6` model for fast and efficient inference.
-   **⚙️ YAML-based Configuration:** Easily manage model settings for different roles (task vs. reflection) via a central configuration file.

---

## 📋 Prerequisites

Before you begin, ensure you have the following:

-   **Python 3.8+** installed on your system.
-   **Pip** for installing Python packages.
-   A **Cerebras API key**. You can get one from the [Cerebras console](https://inference-docs.cerebras.ai/index.html).

---

## 🛠️ Installation and Setup

Follow these steps to set up the project environment.

**1. Clone the Project**
   Download or clone the project files to your local machine.

**2. Set Up a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

**3. Install Dependencies**
   Navigate to the project directory and install the required libraries. Note that `sentence-transformers` and `PyYAML` are installed automatically as dependencies.
   ```bash
   cd gepa_self_optimizer
   pip install dspy-ai
   ```

**4. Set Environment Variable**
   Set your Cerebras API key as an environment variable. The optimizer will not run without it.
   ```bash
   export CEREBRAS_API_KEY="your_actual_cerebras_api_key"
   ```
   *(On Windows, use `set CEREBRAS_API_KEY="your_actual_cerebras_api_key"`)*

---

## 📂 Project Structure

The project is organized into four main files and one configuration directory:

```
gepa_self_optimizer/
├── config/
│   ├── __init__.py
│   └── models.yaml      # Model configurations for Cerebras
├── config.py            # Sets up LMs and the JUDGE_CONSTITUTION
├── system.py            # Defines the Figure-8 self-reflection module
├── generate_data.py     # Script to create synthetic training data
├── optimize_gepa.py     # The main script to run the GEPA optimizer
└── README.md            # This file
```

---

## 🚀 How to Run the Optimizer

The process is split into two main stages: first, you generate the synthetic data, and second, you run the optimizer on that data.

### Step 1: Generate Training Data

The `generate_data.py` script creates a `golden_set.json` file. This file contains question-answer pairs where a deliberate error has been injected. GEPA uses this "failure" data to learn how to write better prompts that avoid such errors.

Run the following command in your terminal:

```bash
python generate_data.py
```

**Expected Output:**
You will see output as the script "sabotages" data with different types of errors. Once complete, you will find a `golden_set.json` file in the `gepa_self_optimizer/` directory.

### Step 2: Run the GEPA Optimizer

This is the core of the project. The `optimize_gepa.py` script loads the `golden_set.json`, defines a metric for success, and then uses the `reflection_lm` to iteratively improve the prompts within the `GlmSelfReflect` system.

Run the following command:

```bash
python optimize_gepa.py
```

**What Happens:**

1.  The script will load the semantic similarity model and the training data.
2.  GEPA will start its evolution cycle. It will run the system, identify failed traces, and use the reflection model to propose and test new prompt instructions.
3.  The process will run for a budget determined by the `auto="medium"` setting in the script.
4.  Upon completion, it will save the final optimized program to `glm_gepa_complete.json`.

---

## 📊 Understanding the Results

After `optimize_gepa.py` finishes, look at your terminal output. You will see a section like this:

```
🏆 GEPA EVOLUTION COMPLETE! Saved to 'glm_gepa_complete.json'

--- What changed? ---
Inspect your optimized program's prompts:
```

This will be followed by the "before and after" or just the final, evolved prompts for the `critic` and `refiner` modules. **Reading these is the best way to understand GEPA's power**, as you can see exactly how it has rewritten the initial instructions to become more effective.

## 🔋 Using Your Optimized Program

The `glm_gepa_complete.json` file contains the `GlmSelfReflect` module with GEPA-optimized prompts. You can load this program and use it for new inference tasks.

Follow these steps in a new Python script or an interactive session:

1.  **Setup your environment and DSPy configuration.**
2.  **Load the optimized program from the JSON file.**
3.  **Run a prediction with a new question.**

**Example Usage Script:**

```python
import dspy
import json
from config import load_config  # Loads task_lm, reflection_lm, etc.
from system import GlmSelfReflect

# 1. Set up the language model context
# Make sure your CEREBRAS_API_KEY is set in the environment
load_config()

# 2. Load the optimized program from the JSON file
optimized_program_path = 'glm_gepa_complete.json'
with open(optimized_program_path, 'r') as f:
    optimized_module = dspy.Program.load_program(f)

print("✅ Optimized GlmSelfReflect program loaded.")

# 3. Run a prediction with a new question
new_question = "Explain the process of photosynthesis in simple terms."
with dspy.context(lm=dspy.settings.lm):
    result = optimized_module(question=new_question)

print(f"\nQuestion: {new_question}")
print(f"Optimized Answer: {result.answer}")
```

This script demonstrates how to integrate your newly optimized model directly into your workflow.

## ⚙️ Configuration

You can modify the model parameters used by the project by editing the `gepa_self_optimizer/config/models.yaml` file. For example, you can adjust the `temperature` or `max_tokens` for either the `task_model` or the `reflection_model`. The system will automatically load these new settings on the next run.

```yaml
# Example snippet from config/models.yaml
# You can adjust the parameters for the task and reflection models here.
# Changes will be picked up the next time you run the optimizer.

task_model:
  name: "zai-glm-4.6"          # Model identifier on Cerebras
  provider: "cerebras"         # The model provider
  temperature: 0.9             # Controls randomness. Higher = more creative.
  max_tokens: 1500             # The maximum number of tokens to generate.

reflection_model:
  name: "zai-glm-4.6"          # Often the same model is used, but you could use a more powerful one.
  provider: "cerebras"
  temperature: 0.5             # Lower temperature is often better for analysis and instruction generation.
  max_tokens: 1000
```
# GEPA Self-Optimizer

A hands-on example of using DSPy's **GEPA (Genetic Evolutionary Prompt Algorithm)** to automatically evolve and optimize the prompts of a Figure-8 self-reflection AI system.

This project demonstrates a powerful prompt engineering technique where a stronger "Reflection" model analyzes failures and proposes textual improvements to a "Task" model's prompts, creating a system that learns to be better over time.

---

## ✨ Features

-   **🧬 Genetic Evolutionary Prompt Algorithm (GEPA):** Leverages DSPy's built-in GEPA optimizer for automated prompt discovery and improvement.
-   **🔄 Figure-8 Self-Reflection Architecture:** Implements a cycle of generation, critique, and refinement to improve answer quality.
-   **🚀 Cerebras Integration:** Optimized to use the Cerebras API with the `zai-glm-4.6` model for fast and efficient inference.
-   **⚙️ YAML-based Configuration:** Easily manage model settings for different roles (task vs. reflection) via a central configuration file.

---

## 📋 Prerequisites

Before you begin, ensure you have the following:

-   **Python 3.8+** installed on your system.
-   **Pip** for installing Python packages.
-   A **Cerebras API key**. You can get one from the [Cerebras console](https://inference-docs.cerebras.ai/index.html).

---

## 🛠️ Installation and Setup

Follow these steps to set up the project environment.

**1. Clone the Project**
   Download or clone the project files to your local machine.

**2. Set Up a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

**3. Install Dependencies**
   Navigate to the project directory and install the required libraries. Note that `sentence-transformers` and `PyYAML` are installed automatically as dependencies.
   ```bash
   cd gepa_self_optimizer
   pip install dspy-ai
   ```

**4. Set Environment Variable**
   Set your Cerebras API key as an environment variable. The optimizer will not run without it.
   ```bash
   export CEREBRAS_API_KEY="your_actual_cerebras_api_key"
   ```
   *(On Windows, use `set CEREBRAS_API_KEY="your_actual_cerebras_api_key"`)*

---

## 📂 Project Structure

The project is organized into four main files and one configuration directory:

```
gepa_self_optimizer/
├── config/
│   ├── __init__.py
│   └── models.yaml      # Model configurations for Cerebras
├── config.py            # Sets up LMs and the JUDGE_CONSTITUTION
├── system.py            # Defines the Figure-8 self-reflection module
├── generate_data.py     # Script to create synthetic training data
├── optimize_gepa.py     # The main script to run the GEPA optimizer
└── README.md            # This file
```

---

## 🚀 How to Run the Optimizer

The process is split into two main stages: first, you generate the synthetic data, and second, you run the optimizer on that data.

### Step 1: Generate Training Data

The `generate_data.py` script creates a `golden_set.json` file. This file contains question-answer pairs where a deliberate error has been injected. GEPA uses this "failure" data to learn how to write better prompts that avoid such errors.

Run the following command in your terminal:

```bash
python generate_data.py
```

**Expected Output:**
You will see output as the script "sabotages" data with different types of errors. Once complete, you will find a `golden_set.json` file in the `gepa_self_optimizer/` directory.

### Step 2: Run the GEPA Optimizer

This is the core of the project. The `optimize_gepa.py` script loads the `golden_set.json`, defines a metric for success, and then uses the `reflection_lm` to iteratively improve the prompts within the `GlmSelfReflect` system.

Run the following command:

```bash
python optimize_gepa.py
```

**What Happens:**

1.  The script will load the semantic similarity model and the training data.
2.  GEPA will start its evolution cycle. It will run the system, identify failed traces, and use the reflection model to propose and test new prompt instructions.
3.  The process will run for a budget determined by the `auto="medium"` setting in the script.
4.  Upon completion, it will save the final optimized program to `glm_gepa_complete.json`.

---

## 📊 Understanding the Results

After `optimize_gepa.py` finishes, look at your terminal output. You will see a section like this:

```
🏆 GEPA EVOLUTION COMPLETE! Saved to 'glm_gepa_complete.json'

--- What changed? ---
Inspect your optimized program's prompts:
```

This will be followed by the "before and after" or just the final, evolved prompts for the `critic` and `refiner` modules. **Reading these is the best way to understand GEPA's power**, as you can see exactly how it has rewritten the initial instructions to become more effective.

## 🔋 Using Your Optimized Program

The `glm_gepa_complete.json` file contains the `GlmSelfReflect` module with GEPA-optimized prompts. You can load this program and use it for new inference tasks.

Follow these steps in a new Python script or an interactive session:

1.  **Setup your environment and DSPy configuration.**
2.  **Load the optimized program from the JSON file.**
3.  **Run a prediction with a new question.**

**Example Usage Script:**

```python
import dspy
import json
from config import load_config  # Loads task_lm, reflection_lm, etc.
from system import GlmSelfReflect

# 1. Set up the language model context
# Make sure your CEREBRAS_API_KEY is set in the environment
load_config()

# 2. Load the optimized program from the JSON file
optimized_program_path = 'glm_gepa_complete.json'
with open(optimized_program_path, 'r') as f:
    optimized_module = dspy.Program.load_program(f)

print("✅ Optimized GlmSelfReflect program loaded.")

# 3. Run a prediction with a new question
new_question = "Explain the process of photosynthesis in simple terms."
with dspy.context(lm=dspy.settings.lm):
    result = optimized_module(question=new_question)

print(f"\nQuestion: {new_question}")
print(f"Optimized Answer: {result.answer}")
```

This script demonstrates how to integrate your newly optimized model directly into your workflow.

## ⚙️ Configuration

You can modify the model parameters used by the project by editing the `gepa_self_optimizer/config/models.yaml` file. For example, you can adjust the `temperature` or `max_tokens` for either the `task_model` or the `reflection_model`. The system will automatically load these new settings on the next run.

```yaml
# Example snippet from config/models.yaml
# You can adjust the parameters for the task and reflection models here.
# Changes will be picked up the next time you run the optimizer.

task_model:
  name: "zai-glm-4.6"          # Model identifier on Cerebras
  provider: "cerebras"         # The model provider
  temperature: 0.9             # Controls randomness. Higher = more creative.
  max_tokens: 1500             # The maximum number of tokens to generate.

reflection_model:
  name: "zai-glm-4.6"          # Often the same model is used, but you could use a more powerful one.
  provider: "cerebras"
  temperature: 0.5             # Lower temperature is often better for analysis and instruction generation.
  max_tokens: 1000
```
# GEPA Self-Optimizer

A hands-on example of using DSPy's **GEPA (Genetic Evolutionary Prompt Algorithm)** to automatically evolve and optimize the prompts of a Figure-8 self-reflection AI system.

This project demonstrates a powerful prompt engineering technique where a stronger "Reflection" model analyzes failures and proposes textual improvements to a "Task" model's prompts.

---

## ✨ Features

-   **🧬 Genetic Evolutionary Prompt Algorithm (GEPA):** Leverages DSPy's built-in GEPA optimizer for automated prompt discovery.
-   **🔄 Figure-8 Self-Reflection Architecture:** Implements a cycle of generation, critique, and refinement.
-   **⚙️ YAML-based Configuration:** Easily manage models and run parameters via configuration files.
-   **📊 Configurable Runs:** Switch between a quick "test run" and a full "optimization run" without changing code.

---

## 🛠️ Installation and Setup

**1. Prerequisites**
-   Python 3.8+
-   Pip for installing packages.
-   An API key for your chosen LM provider (e.g., Cerebras, ZhipuAI).

**2. Install Dependencies**
```bash
pip install dspy-ai sentence-transformers pyyaml torch
```

**3. Set Environment Variable**
Set your API key as an environment variable.
```bash
export CEREBRAS_API_KEY="your_actual_cerebras_api_key"
# OR
export ZHIPUAI_API_KEY="your_actual_zhipuai_api_key"
```

## 📂 Configuration

The project is configured using YAML files in `gepa_self_optimizer/config/`:

-   `models.yaml`: Define your task and reflection models. A `models.dev.yaml` is also provided for faster, cheaper testing.
-   `settings.yaml`: Control run parameters like the number of data examples and the optimization intensity.

## 🚀 Usage

### Quick Test Run

This configuration performs a fast, inexpensive run to test your setup.

**1. Configure for a small run in `settings.yaml`:**
```yaml
data_generation:
  num_examples: 2
optimization:
  gepa_auto_setting: "light"
```

**2. Execute the workflow:**
```bash
python gepa_self_optimizer/generate_data.py
python gepa_self_optimizer/optimize_gepa.py
```

### Full Optimization Run

For a more thorough optimization, use the default settings in `settings.yaml`:
```yaml
data_generation:
  num_examples: 25
optimization:
  gepa_auto_setting: "medium"
```
Then, run the same two commands as above.

---

## 📊 Understanding the Results

-   `golden_set.json`: The generated QA pairs with injected errors, used as training data.
-   `glm_gepa_complete.json`: The final, optimized `GlmSelfReflect` program. You can load and use this program for new inference tasks.
-   The console output will display the optimized prompts, showing you how GEPA improved the original instructions.

## 🔋 Using Your Optimized Program

You can load the saved, optimized program and run it on new questions:

```python
import dspy
import json
from gepa_config import setup_dspy
from gepa_system import GlmSelfReflect

# 1. Set up the language model context
load_config()

# 2. Load the optimized program from the JSON file
with open("glm_gepa_complete.json", "r") as f:
    optimized_module = dspy.Program.load_program(f)

print("✅ Optimized GlmSelfReflect program loaded.")

# 3. Run a prediction with a new question
new_question = "Explain the process of photosynthesis in simple terms."
with dspy.context(lm=dspy.settings.lm):
    result = optimized_module(question=new_question)

print(f"\nQuestion: {new_question}")
print(f"Optimized Answer: {result.answer}")
```
