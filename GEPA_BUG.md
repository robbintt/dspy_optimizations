# [BUG] dspy.GEPA failing to learn and save demonstrations

**Description:**

I am encountering a bug with `dspy.GEPA` where the optimization loop completes successfully, but the final optimized program returned is completely empty of learned demonstrations (`demos`).

**Environment:**

*   `dspy` version: 3.0.4 (from your `metadata.json`)
*   `python` version: 3.12

**Steps to Reproduce:**

1.  Define a multi-predictor `dspy.Module` (e.g., `GlmSelfReflect` with `generator`, `critic`, `refiner`).
2.  Create a training set and a metric function that returns a `dspy.Prediction` with a `score` and `feedback`.
3.  Configure and run `dspy.GEPA` with `track_stats=True`. For example:
    ```python
    optimizer = dspy.GEPA(metric=MyMetric, auto="light", ...)
    optimized_program = optimizer.compile(student=MyProgram, trainset=trainset, valset=valset)
    ```
4.  Inspect the optimized program, either by checking `optimized_program.critic.demos` or by saving it to a file.

**Expected Behavior:**

The `optimized_program` should contain its predictors populated with learned demonstrations based on the optimization process.

**Actual Behavior:**

The optimization loop runs, the metric is called, and scores are tracked. However, the final `optimized_program` is returned with `demos` lists that are empty for all predictors.
*   `optimized_program.critic.demos` is `[]`
*   `optimized_program.refiner.demos` is `[]`
*   etc.

This was confirmed by inspecting the final saved JSON file, where every predictor has `"demos": []`.

**Additional Context:**

I have performed extensive debugging to rule out other causes:
*   The metric is being called and returning valid, non-zero scores.
*   The training and validation sets are loaded correctly and are non-empty.
*   The issue persists even when simplifying the module to use only `dspy.Predict` predictors, ruling out an interaction with `dspy.ChainOfThought`.

The problem appears to be that the GEPA optimizer is not correctly transferring the learned demonstrations from its internal best candidate back into the final program object it constructs and returns.

**Attachments:**

I have attached the final saved program state file, which clearly shows the empty `demos` arrays throughout.
*   `glm_gepa_complete.json`
