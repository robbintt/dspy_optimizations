# GEPA Self-Optimizer Fixes

This document outlines critical fixes needed for the `gepa_self_optimizer` module to run correctly.

---

## 1. Critical Initialization Error

### Problem
`task_lm` and `reflection_lm` are `None` at runtime, which causes the application to crash on startup.

### Location
- `gepa_config.py` (Root cause)
- `generate_data.py`
- `optimize_gepa.py`

### Explanation
The language models are initialized to `None` in `gepa_config.py`. The `setup_dspy()` function is responsible for their actual initialization, but this function is never called in `generate_data.py` or `optimize_gepa.py`.

### Solution
Import and call `setup_dspy()` at the beginning of each script before using any dspy models.

**In `generate_data.py`:**
```python
from gepa_config import setup_dspy, task_lm, _load_run_settings

# Initialize DSPy and models first
setup_dspy()

# Load run settings
run_settings = _load_run_settings()

# Now task_lm is initialized
with dspy.context(lm=task_lm):
    # ...
```

**In `optimize_gepa.py`:**
```python
from gepa_config import setup_dspy

# Initialize DSPy and models first
setup_dspy()

# Now proceed with optimization
optimizer = dspy.GEPA(...)
```

---

## 2. API Key Configuration Mismatch

### Problem
The configuration is hardcoded to look for a `CEREBRAS_API_KEY`, which will fail if you are using a different provider like ZhipuAI.

### Location
- `gepa_config.py`

### Explanation
Hardcoding a specific environment variable name makes the configuration inflexible. If your `models.yaml` points to a ZhipuAI model, the script will still look for the incorrect API key.

### Solution
Update the environment variable name in `gepa_config.py` to match your provider.

**Example for ZhipuAI:**
In `setup_dspy()`, change:
```python
# From
api_key = os.getenv("CEREBRAS_API_KEY")

# To
api_key = os.getenv("ZHIPUAI_API_KEY")
```

---

## 3. Incompatible GEPA Optimizer Parameters

### Problem
The `dspy.GEPA` optimizer is being passed parameters meant for few-shot optimizers.

### Location
- `optimize_gepa.py`

### Explanation
`dspy.GEPA` is an **instruction optimizer**. It refines prompts but does not handle few-shot example generation. The `max_bootstrapped_demos` and `max_labeled_demos` parameters are used by optimizers like `MIPROv2`. Passing them to `GEPA` may be ignored or cause a `TypeError` depending on the DSPy version.

### Solution
Remove the incompatible parameters from the `dspy.GEPA` constructor.

**In `optimize_gepa.py`:**
Change:
```python
optimizer = dspy.GEPA(
    metric=refinement_gepa_metric,
    auto="medium",
    reflection_lm=reflection_lm,
    track_stats=True,
    max_bootstrapped_demos=4,      # <- REMOVE
    max_labeled_demos=16           # <- REMOVE
)
```

To:
```python
# Get the GEPA auto setting from settings, with a default of "medium"
gepa_auto_setting = run_settings.get("optimization", {}).get("gepa_auto_setting", "medium")

optimizer = dspy.GEPA(
    metric=refinement_gepa_metric,
    auto=gepa_auto_setting,
    reflection_lm=reflection_lm,
    track_stats=True
)
```
