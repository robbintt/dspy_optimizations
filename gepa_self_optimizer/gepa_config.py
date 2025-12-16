import os
import yaml
from pathlib import Path
# Import main dspy module at runtime (will be mocked in tests)
try:
    import dspy
except ImportError:
    dspy = None
# from dspy.primitives import Signature, InputField, OutputField

# Initialize module-level variables to None so they can be imported
lm = None
task_lm = None
reflection_lm = None

# --- 1. LOAD CONFIGURATIONS FROM YAML ---
# Determine the directory of this script to find the config file
# Use absolute path to ensure it works during pytest discovery
CONFIG_DIR = Path(__file__).parent.resolve()
MODEL_CONFIG_PATH = CONFIG_DIR / "config" / "models.yaml"

# Load the entire model configuration file
# Make this lazy loading to allow for test mocking
def _load_model_configs():
    with open(MODEL_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def _create_lm(config_name: str):
    """
    Helper function to create a dspy.LM instance from a named configuration.
    """
    # Load configs directly inside the function
    all_configs = _load_model_configs()
    gepa_config = all_configs.get(config_name)
    if not gepa_config:
        raise ValueError(f"Model configuration '{config_name}' not found in {MODEL_CONFIG_PATH}")
    
    # Construct the model identifier in 'provider/model_name' format
    model_id = f"{gepa_config['provider']}/{gepa_config['name']}"
    
    # Extract parameters that are not the model identifier itself
    lm_params = {k: v for k, v in gepa_config.items() if k not in ['provider', 'name']}
    
    # Determine the appropriate API key environment variable based on provider
    provider = gepa_config['provider'].lower()
    if provider == "cerebras":
        api_key_env = "CEREBRAS_API_KEY"
    elif provider == "zhipuai":
        api_key_env = "ZHIPUAI_API_KEY"
    elif provider == "openai":
        api_key_env = "OPENAI_API_KEY"
    elif provider == "anthropic":
        api_key_env = "ANTHROPIC_API_KEY"
    else:
        # Default to CEREBRAS_API_KEY for backwards compatibility
        api_key_env = "CEREBRAS_API_KEY"
    
    api_key = os.getenv(api_key_env)
    return dspy.LM(model=model_id, api_key=api_key, **lm_params)

def setup_dspy(api_key: str = None):
    """
    Initializes and configures the dspy language models.
    MUST be called explicitly in the main application script.
    It will NOT run during pytest discovery.
    """
    # Check if any of the supported API keys are set
    api_key_vars = ["CEREBRAS_API_KEY", "ZHIPUAI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    api_key_found = False
    
    if api_key is not None:
        # Use explicitly provided API key
        os.environ["DSPY_API_KEY"] = api_key
        api_key_found = True
    else:
        # Check for any supported API key in environment
        for var in api_key_vars:
            if os.getenv(var):
                api_key_found = True
                break
    
    if not api_key_found:
        raise ValueError(
            "Please set one of the following environment variables: " + 
            ", ".join(api_key_vars)
        )

    # --- 3. INSTANTIATE THE LANGUAGE MODELS ---
    global task_lm, reflection_lm, lm  # Declare we're modifying globals
    task_lm = _create_lm("task_model")
    reflection_lm = _create_lm("reflection_model")

    # --- 4. CONFIGURE DSPY ---
    # The main lm for DSPy operations will be the task_lm
    lm = task_lm
    if dspy is not None:
        dspy.configure(lm=lm)


# --- 4. THE JUDGE'S CONSTITUTION ---
JUDGE_CONSTITUTION = """
You are a Constitutional Critic. Adhere to these principles:
1. FALSEHOODS ARE FATAL: If an answer contains a factual error, mark it INVALID.
2. NO SYCOPHANCY: Do not be polite. Be pedantic.
3. CODE MUST RUN: In code tasks, syntax errors are immediate failures.
4. LOGIC OVER STYLE: Ignore tone; focus on the reasoning chain.
"""
