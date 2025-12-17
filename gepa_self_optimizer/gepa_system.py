import dspy
import json
from dspy.primitives.base_module import BaseModule
from gepa_config import JUDGE_CONSTITUTION, create_gepa_optimizer
import json

def post_compile_inspection(program, program_name="Optimized Program"):
    """
    Inspects a program immediately after optimization.
    This helps determine if the optimizer or the save function is the source of corruption.
    """
    print(f"\n🐛 [DEBUG] Inspecting '{program_name}' immediately after GEPA compilation...")
    if not hasattr(program, 'dump_state'):
        print(f"  ⚠️ Program {program_name} does not have a dump_state method. Cannot inspect.")
        return

    state = program.dump_state()
    for component_key, component_data in state.items():
        # Handle nested state for ChainOfThought modules, which have a '.predict' suffix
        # and an internal 'predict' key in their dumped state.
        if 'predict' in component_data:
            actual_data = component_data.get('predict', {})
        else:
            actual_data = component_data
            
        demos = actual_data.get('demos', [])
        instructions = actual_data.get('signature', {}).get('instructions', '')
        print(f"  🔍 Component: {component_key}")
        print(f"    -> 'demos' count: {len(demos)}")
        print(f"    -> 'instructions' length: {len(instructions)}")
        # To avoid huge log output, we won't print the full instructions here unless needed.
    print(f"🐛 [DEBUG] Finished post-compilation inspection of '{program_name}'.\n")


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
        # WORKAROUND: Use dspy.Predict for generator to avoid a GEPA bug with ChainOfThought state.
        # https://github.com/stanfordnlp/dspy/issues/XXX (placeholder)
        self.generator = dspy.Predict(Generate)
        self.critic = dspy.ChainOfThought(ShepherdCritic)
        self.refiner = dspy.Predict(Refine)

    def predictors(self):
        return [self.generator, self.critic, self.refiner]

    def dump_state(self, json_mode=True):
        """
        An instrumented version of dump_state to debug the saving process.
        It prints the state of each component before saving.
        """
        print("\n🐛 [DEBUG] Starting instrumented dump_state...")
        
        state = {}
        # Iterate through items stored in the module's __dict__
        for name, value in self.__dict__.items():
            # We only care about dspy modules (predictors)
            if isinstance(value, BaseModule):
                print(f"  🔍 Inspecting component: '{name}' of type {type(value)}")
                
                # Get the core state dict of the predictor (instructions, demos, etc.)
                predictor_state = value.dump_state(json_mode=json_mode)
                print(f"    -> 'demos' count: {len(predictor_state.get('demos', []))}")
                print(f"    -> 'instructions' length: {len(predictor_state.get('signature', {}).get('instructions', ''))}")
                
                state[f"{name}.predict"] = predictor_state
            else:
                print(f"  ⏭️ Skipping non-module attribute: '{name}' ({type(value)})")

        print("🐛 [DEBUG] Finished instrumented dump_state. Saving the following state.")
        # print(json.dumps(state, indent=2)) # Optionally print the full state

        return state

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
