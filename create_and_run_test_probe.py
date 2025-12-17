import dspy
import json

# Define a simple signature for the test
class TestSignature(dspy.Signature):
    """This is the docstring and default instruction for the signature."""
    input_str = dspy.InputField(desc="Input string.")
    output_str = dspy.OutputField(desc="Output string.")

def test_chain_of_thought_state_bug():
    """
    Probes for a potential bug in dspy.ChainOfThought's state management.
    """
    print("--- Probing dspy.ChainOfThought for state bug ---")

    # Step 1: Create a ChainOfThought predictor with an initial instruction.
    initial_instruction = "This is the EXPLICIT initial instruction."
    predictor = dspy.ChainOfThought(TestSignature, instructions=initial_instruction)
    
    # Verify the initial state. ChainOfThought stores the instruction in the signature of its internal predictor.
    current_instruction_after_init = predictor.predict.signature.instructions
    print(f"1. Explicit instruction set: '{initial_instruction}'")
    print(f"   Module reports after init: '{current_instruction_after_init}'")
    print(f"   -> Explicit instruction used? {initial_instruction == current_instruction_after_init}\n")

    if initial_instruction != current_instruction_after_init:
        print("   NOTE: The explicit instruction was ignored and the docstring was used. This is a separate issue.\n")


    # Step 2: Manually mutate the instruction inside the module.
    new_instruction = "THIS IS A NEW MUTATED INSTRUCTION TO TEST BUG."
    print(f"2. Manually updating instruction to: '{new_instruction}'")
    predictor.predict.signature.instructions = new_instruction
    print(f"   Update complete.\n")
    

    # Step 3: Call dump_state() to see if it reflects the change.
    print("3. Calling dump_state() to inspect the saved state...")
    dumped_state = predictor.dump_state()

    # Step 4: Analyze the result. The state is nested under the 'predict' key.
    print("   -> Raw dump_state keys:", list(dumped_state.keys()))
    predictor_state = dumped_state.get('predict', {})

    if predictor_state and 'signature' in predictor_state and 'instructions' in predictor_state['signature']:
        instruction_from_dump_state = predictor_state['signature']['instructions']
        print(f"   -> Instruction from dump_state()['predict']['signature']['instructions']: '{instruction_from_dump_state}'")
        
        if instruction_from_dump_state == new_instruction:
            print("\n✅ SUCCESS: dump_state() correctly reflects the mutated instruction. No bug detected here.")
            return True
        else:
            print("\n❌ FAILURE: dump_state() shows the OLD (or default) instruction. This confirms a bug in dspy.ChainOfThought.")
            print(f"   Expected: '{new_instruction}'")
            print(f"   Got:      '{instruction_from_dump_state}'")
            return False
    else:
        print("\n❌ ERROR: Could not find nested 'instructions' key in the dumped state.")
        print("   Predictor state keys:", list(predictor_state.keys()))
        return False

if __name__ == "__main__":
    # Configure a dummy LM to prevent errors
    with dspy.context(lm=dspy.LM(model="dummy_model", api_key="dummy_key")):
        test_chain_of_thought_state_bug()
import dspy
import json

# --- Setup a minimal environment ---
# Use a dummy LM to avoid API calls during the probe
dummy_lm = dspy.LM(model="dummy_model", api_key="dummy_key")

class SimpleChainOfThoughtTask(dspy.Signature):
    """Signature for the ChainOfThought predictor."""
    question = dspy.InputField(desc="A simple question.")
    answer = dspy.OutputField(desc="A simple answer.")

class SimplePredictTask(dspy.Signature):
    """Signature for the Predict predictor."""
    question = dspy.InputField(desc="A simple question.")
    answer = dspy.OutputField(desc="A simple answer.")

def simple_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    A simple metric that always returns 1.0 to ensure success.
    It matches the required 5-argument signature for dspy.GEPA.
    """
    return 1.0

# --- 2. Define a more realistic student program ---
class SimpleProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        # Module structure resembles GlmSelfReflect
        self.chain_of_thought_predictor = dspy.ChainOfThought(
            SimpleChainOfThoughtTask, 
            instructions="This is the initial instruction for the ChainOfThought predictor."
        )
        self.predict_predictor = dspy.Predict(
            SimplePredictTask,
            instructions="This is the initial instruction for the Predict predictor."
        )
    
    def forward(self, question):
        # Predictors need to be named to be found by GEPA
        with dspy.context(predictor=self.chain_of_thought_predictor):
            cot_output = self.chain_of_thought_predictor(question=question)
        
        with dspy.context(predictor=self.predict_predictor):
            pred_output = self.predict_predictor(question=question)

        # Use the output from one of the predictors
        return dspy.Prediction(answer=cot_output.answer)

def test_gepa_compilation_bug():
    """
    Probes for a bug inside dspy.GEPA's compilation process that corrupts
    the state of ChainOfThought modules.
    """
    print("--- Probing dspy.GEPA for a compilation bug ---")

    with dspy.context(lm=dummy_lm):
        # --- 3. Create the student and optimizer ---
        student = SimpleProgram()
        
        # Sanity check: Manually mutate and verify student is OK before GEPA
        print("1. Pre-GEPA sanity check:")
        student.complex_predictor.predict.signature.instructions = "MANUALLY MUTATED INSTRUCTION"
        state_before_gepa = student.dump_state()
        
        # The state of a module is a dict of its components. We need to access the one we care about.
        complex_predictor_state = state_before_gepa.get('complex_predictor', {})
        instruction_before_gepa = complex_predictor_state.get('predict', {}).get('signature', {}).get('instructions')
        
        print(f"   -> Student instruction before GEPA: '{instruction_before_gepa}'\n")

        # GEPA needs a trainset, even if it's just one example
        trainset = [dspy.Example(question="Q", answer="A").with_inputs("question")]
        
        # Use a minimal GEPA config. A single metric call is enough for the probe.
        # GEPA requires a reflection_lm, so we provide a dummy one.
        optimizer = dspy.GEPA(metric=simple_metric, max_metric_calls=1, track_stats=False, reflection_lm=dummy_lm)

        # --- 4. The critical step: Compile the program ---
        print("2. Running optimizer.compile()...")
        try:
            optimized_program = optimizer.compile(student=student, trainset=trainset)
            print("   -> Compilation finished without errors.\n")
        except Exception as e:
            print(f"   -> Compilation FAILED with error: {e}\n")
            return False

        # --- 5. Inspect the program returned by the optimizer ---
        print("3. Inspecting the program returned by GEPA:")
        
        # Reset instruction to a known value to see what GEPA did
        optimized_program.complex_predictor.predict.signature.instructions = "VALUE SET AFTER GEPA"
        
        state_after_gepa = optimized_program.dump_state()
        
        # The state of a module is a dict of its components. We need to access the one we care about.
        complex_predictor_state_after = state_after_gepa.get('complex_predictor', {})
        
        if not complex_predictor_state_after:
            print("   -> ❌ FAILURE: Optimized program state is missing its 'complex_predictor' component.")
            print("   -> Full state:", state_after_gepa)
            return False

        instruction_after_gepa = complex_predictor_state_after.get('predict', {}).get('signature', {}).get('instructions')
        print(f"   -> Optimized program instruction: '{instruction_after_gepa}'")

        # --- 6. Analyze the result ---
        if instruction_after_gepa == "":
            print("\n   -> ❌ FAILURE CONFIRMED: GEPA returned a ChainOfThought module with a corrupted, empty instruction.")
            print("      This confirms the bug is within the GEPA optimizer itself.")
            return False
        else:
            print("\n   -> ✅ SUCCESS: The optimized program has a valid instruction. GEPA compilation did not corrupt the state.")
            return True

if __name__ == "__main__":
    test_gepa_compilation_bug()
