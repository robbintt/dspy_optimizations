import dspy
import json
from dspy.primitives.base_module import BaseModule
from gepa_config import JUDGE_CONSTITUTION, create_gepa_optimizer
import json

def post_compile_inspection(program, program_name="Optimized Program"):
    """
    Inspects a program immediately after optimization.
    This is now a no-op since GEPA is working correctly.
    """
    pass


def optimize_with_retries(student_module, trainset, valset, reflection_lm, metric, config, max_retries=3):
    """
    Wrapper function to run GEPA optimization with retries on reflection failures.
    
    Args:
        student_module: The DSPy module to optimize
        trainset: Training data for optimization
        valset: Validation data for evaluation
        reflection_lm: The language model to use for reflection
        metric: The metric function for evaluation
        config: The GEPARunConfig to use (MUST be provided)
        max_retries: Maximum number of retry attempts on reflection failures
        
    Returns:
        The optimized DSPy module
        
    Raises:
        ValueError: If config is None
    """
    if config is None:
        raise ValueError("Configuration must be explicitly provided. Use a config from gepa_config or create your own GEPARunConfig instance.")
    # Create optimizer from configuration
    gepa_optimizer = create_gepa_optimizer(
        metric=metric,
        config=config,
        reflection_lm=reflection_lm
    )
    
    optimized_program = None
    
    if config.max_metric_calls:
        print(f"Starting GEPA optimization with budget of {config.max_metric_calls} calls")
    else:
        print(f"Starting GEPA optimization with up to {max_retries} retries...")
    
    for attempt in range(max_retries):
        try:
            print(f"--- Attempt {attempt + 1}/{max_retries} ---")
            optimized_program = gepa_optimizer.compile(
                student=student_module,
                trainset=trainset,
                valset=valset,
            )
            print("GEPA compilation successful.")
            break  # Exit the retry loop on success
            
        except Exception as e:
            if "No valid predictions found for any module." in str(e):
                print(f"Attempt {attempt + 1} failed with a reflection error. Will retry.")
            else:
                print(f"Attempt {attempt + 1} failed with an unexpected error: {e}")
                raise
    
    if optimized_program is None:
        raise RuntimeError(f"GEPA compilation failed after {max_retries} retries.")
    
    print("GEPA optimization finished successfully.")
    
    # Inspect the program returned by the optimizer *before* passing it to the caller
    post_compile_inspection(optimized_program, program_name="GEPA Optimized Program")
    
    return optimized_program

# --- SIGNATURES ---
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

# --- THE MODULE ---
class GlmSelfReflect(dspy.Module):
    def __init__(self):
        super().__init__()
        # CHANGING TO ALL PREDICT MODULES TO ISOLATE THE GEPA+ChainOfThought BUG
        self.generator = dspy.Predict(Generate)
        self.critic = dspy.Predict(ShepherdCritic) # Changed from ChainOfThought
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
