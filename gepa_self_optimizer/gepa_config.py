import os
import yaml
import dspy
from pathlib import Path

# --- 1. LOAD CONFIGURATIONS FROM YAML ---
# Determine the directory of this script to find the config file
CONFIG_DIR = Path(__file__).parent
MODEL_CONFIG_PATH = CONFIG_DIR / "models.yaml"

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
    gepa_config = model_configs.get(config_name)
    if not gepa_config:
        raise ValueError(f"Model configuration '{config_name}' not found in {MODEL_CONFIG_PATH}")
    
    # Construct the model identifier in 'provider/model_name' format
    model_id = f"{gepa_config['provider']}/{gepa_config['name']}"
    
    # Extract parameters that are not the model identifier itself
    lm_params = {k: v for k, v in gepa_config.items() if k not in ['provider', 'name']}
    
    return dspy.LM(model=model_id, api_key=API_KEY, **lm_params)

# --- 3. INSTANTIATE THE LANGUAGE MODELS ---
# The task model for generating and refining
task_lm = _create_lm("task_model")

# The reflection model for GEPA's self-improvement step
reflection_lm = _create_lm("reflection_model")

# Set the default LM for DSPy to our task model
dspy.configure(lm=task_lm)

# --- 4. THE JUDGE'S CONSTITUTION ---
JUDGE_CONSTITUTION = """
You are a Constitutional Critic. Adhere to these principles:
1. FALSEHOODS ARE FATAL: If an answer contains a factual error, mark it INVALID.
2. NO SYCOPHANCY: Do not be polite. Be pedantic.
3. CODE MUST RUN: In code tasks, syntax errors are immediate failures.
4. LOGIC OVER STYLE: Ignore tone; focus on the reasoning chain.
"""
