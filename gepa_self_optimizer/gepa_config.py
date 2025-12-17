import os
import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
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
    final_config = {}
    with open(MODEL_CONFIG_PATH, "r") as f:
        # safe_load_all creates a generator for all documents in the stream
        for document in yaml.safe_load_all(f):
            if isinstance(document, dict):
                final_config.update(document)
    return final_config

def _load_run_settings():
    """Loads the run-specific settings, including the GEPA profile, from config files."""
    SETTINGS_CONFIG_PATH = CONFIG_DIR / "config" / "settings.yaml"
    
    # Start with model configs to find the gepa_profile
    model_configs = _load_model_configs()
    
    # Load other settings from settings.yaml if it exists
    try:
        with open(SETTINGS_CONFIG_PATH, "r") as f:
            additional_settings = yaml.safe_load(f) or {}
            model_configs.update(additional_settings)
    except FileNotFoundError:
        # Return model configs if the optional config file is not found
        pass
    
    return model_configs

# Load run settings at module level so they can be imported
run_settings = _load_run_settings()

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

    # Validate that models were created successfully
    if task_lm is None:
        raise RuntimeError("Failed to create task language model")
    if reflection_lm is None:
        raise RuntimeError("Failed to create reflection language model")

    # --- 4. CONFIGURE DSPY ---
    # The main lm for DSPy operations will be the task_lm
    lm = task_lm
    if dspy is not None:
        dspy.configure(lm=lm)

    # Return the created language models for direct use
    return task_lm, reflection_lm


# --- 4. SEMANTIC SIMILARITY FUNCTION ---
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

from dataclasses import dataclass
from typing import Optional, Dict, Any
import dspy


def semantic_similarity(text1, text2):
    """Computes cosine similarity between two texts."""
    embeddings = similarity_model.encode([text1, text2], convert_to_tensor=True)
    return util.cos_sim(embeddings[0], embeddings[1]).item()

# --- 5. METRIC FOR GEPA ---
def refinement_gepa_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Computes a semantic similarity score and provides rich feedback for GEPA.
    GEPA uses this function to determine if a prediction was successful and to gather
    textual feedback from its failures to inform prompt improvements.
    """
    score = semantic_similarity(prediction.answer, example.correct_answer)
    
    # The feedback is critical: it tells the reflection model *why* it failed
    # and gives it the 'gold standard' to aim for.
    feedback = (
        f"The generated answer achieved a similarity score of {score:.3f} against the reference answer. "
        f"The model's prediction was:\n---\n{prediction.answer}\n---\n"
        f"The target reference answer was:\n---\n{example.correct_answer}\n---"
    )
    
    # Return a ScoreWithFeedback object, which is the standard for DSPy optimizers.
    # GEPA will try to maximize this score.
    return dspy.evaluate.ScoreWithFeedback(score=score, feedback=feedback)

# --- 6. THE JUDGE'S CONSTITUTION ---
def _load_judge_constitution():
    """Load the judge's constitution from a markdown file."""
    constitution_path = CONFIG_DIR / "JUDGE_CONSTITUTION.md"
    try:
        with open(constitution_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback to a default constitution if file is not found
        return """You are a Constitutional Critic. Adhere to these principles:
1. FALSEHOODS ARE FATAL: If an answer contains a factual error, mark it INVALID.
2. NO SYCOPHANCY: Do not be polite. Be pedantic.
3. CODE MUST RUN: In code tasks, syntax errors are immediate failures.
4. LOGIC OVER STYLE: Ignore tone; focus on the reasoning chain."""

JUDGE_CONSTITUTION = _load_judge_constitution()


@dataclass
class GEPARunConfig:
    """
    Configuration for a GEPA optimization run.
    Each instance represents a complete set of parameters for one optimization job.
    """
    
    # Profile identifier
    gepa_profile: Optional[str] = None
    
    # Budget configuration
    max_metric_calls: Optional[int] = None
    max_full_evals: Optional[int] = None
    auto: Optional[str] = None  # "light", "medium", "heavy"
    
    # Reflection configuration
    reflection_minibatch_size: int = 3
    candidate_selection_strategy: str = "pareto"
    skip_perfect_score: bool = True
    
    # Merge configuration
    use_merge: bool = True
    max_merge_invocations: int = 5
    
    # Evaluation configuration
    num_threads: Optional[int] = None
    failure_score: float = 0.0
    perfect_score: float = 1.0
    
    # Logging configuration
    log_dir: Optional[str] = None
    track_stats: bool = False
    warn_on_score_mismatch: bool = True
    
    # Reproducibility
    seed: int = 0
    
    # Additional GEPA kwargs
    gepa_kwargs: Optional[Dict[str, Any]] = None
    
    def validate(self):
        """Validate the configuration."""
        budget_params = [
            self.max_metric_calls is not None,
            self.max_full_evals is not None,
            self.auto is not None
        ]
        if sum(budget_params) != 1:
            raise ValueError(
                "Exactly one of max_metric_calls, max_full_evals, or auto must be set"
            )


# --- GEPA PROFILE LOADER ---

def get_gepa_run_config(profile_name: str) -> GEPARunConfig:
    """
    Loads a predefined GEPARunConfig object by its string profile name.

    Args:
        profile_name: The string name of the profile (e.g., "development", "medium").
    
    Returns:
        The corresponding GEPARunConfig instance.
    
    Raises:
        ValueError: If the profile_name is not found.
    """
    profile = GEPA_CONFIG_PROFILES.get(profile_name.lower())
    if not profile:
        raise ValueError(
            f"Unknown GEPA profile '{profile_name}'. "
            f"Available profiles: {list(GEPA_CONFIG_PROFILES.keys())}"
        )
    return profile

# Pre-defined configurations for common use cases
DEVELOPMENT_CONFIG = GEPARunConfig(
    gepa_profile="development",
    max_metric_calls=80,
    reflection_minibatch_size=3,
    use_merge=True,
    max_merge_invocations=2,
    track_stats=True,
    warn_on_score_mismatch=False,
    seed=42,
)

LIGHT_CONFIG = GEPARunConfig(
    gepa_profile="small",
    auto="light",
    reflection_minibatch_size=3,
    use_merge=True,
    max_merge_invocations=3,
    track_stats=True,
    seed=42,
)

MEDIUM_CONFIG = GEPARunConfig(
    gepa_profile="medium",
    auto="medium",
    reflection_minibatch_size=3,
    use_merge=True,
    max_merge_invocations=5,
    track_stats=True,
    seed=42,
)

HEAVY_CONFIG = GEPARunConfig(
    gepa_profile="large",
    auto="heavy",
    reflection_minibatch_size=4,
    use_merge=True,
    max_merge_invocations=8,
    track_stats=True,
    seed=42,
)

GEPA_CONFIG_PROFILES = {
    "development": DEVELOPMENT_CONFIG,
    "small": LIGHT_CONFIG,
    "medium": MEDIUM_CONFIG,
    "large": HEAVY_CONFIG,
}


def get_default_gepa_run_config() -> GEPARunConfig:
    """
    Loads the GEPARunConfig specified by the 'gepa_profile' key in the
    model configuration file.
    """
    profile_name = run_settings.get('gepa_profile', 'development')
    if not profile_name:
        raise ValueError(
            "A 'gepa_profile' must be specified in the model configuration "
            "or settings.yaml to get a default config."
        )
    return get_gepa_run_config(profile_name)

def create_gepa_optimizer(metric, config: GEPARunConfig, reflection_lm: dspy.LM) -> dspy.GEPA:
    """
    Create a GEPA optimizer from a configuration.
    
    Args:
        metric: The metric function to use for feedback and evaluation
        config: The GEPARunConfig instance
        reflection_lm: The language model to use for reflection
        
    Returns:
        Configured dspy.GEPA optimizer
    """
    config.validate()
    
    # Prepare the base arguments for dspy.GEPA, excluding budget settings
    optimizer_kwargs = {
        "metric": metric,
        "reflection_lm": reflection_lm,
        "reflection_minibatch_size": config.reflection_minibatch_size,
        "candidate_selection_strategy": config.candidate_selection_strategy,
        "skip_perfect_score": config.skip_perfect_score,
        "use_merge": config.use_merge,
        "max_merge_invocations": config.max_merge_invocations,
        "num_threads": config.num_threads,
        "failure_score": config.failure_score,
        "perfect_score": config.perfect_score,
        "log_dir": config.log_dir,
        "track_stats": config.track_stats,
        "warn_on_score_mismatch": config.warn_on_score_mismatch,
        "seed": config.seed,
        "gepa_kwargs": config.gepa_kwargs or {},
    }
    
    # Add only the relevant budget parameter to avoid conflicts in dspy.GEPA
    if config.auto is not None:
        optimizer_kwargs["auto"] = config.auto
    elif config.max_metric_calls is not None:
        optimizer_kwargs["max_metric_calls"] = config.max_metric_calls
    elif config.max_full_evals is not None:
        optimizer_kwargs["max_full_evals"] = config.max_full_evals
        
    return dspy.GEPA(**optimizer_kwargs)
