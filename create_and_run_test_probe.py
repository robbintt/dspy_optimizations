import dspy
import json

# Define a simple signature for the test
class TestSignature(dspy.Signature):
    """A simple signature for testing state changes."""
    input_str = dspy.InputField(desc="Input string.")
    output_str = dspy.OutputField(desc="Output string.")

def test_chain_of_thought_state_bug():
    """
    Probes for a potential bug in dspy.ChainOfThought's state management.
    """
    print("--- Probing dspy.ChainOfThought for state bug ---")

    # Step 1: Create a ChainOfThought predictor with an initial instruction.
    initial_instruction = "This is the initial instruction."
    predictor = dspy.ChainOfThought(TestSignature, instructions=initial_instruction)
    
    # Sanity check: Verify the initial state via its internal attribute
    # Note: The instruction is stored inside the signature within the 'predict' attribute
    current_instruction_after_init = predictor.predict.signature.instructions
    print(f"1. Initial instruction set: '{initial_instruction}'")
    print(f"   Module reports instruction after init: '{current_instruction_after_init}'")
    print(f"   -> Match? {initial_instruction == current_instruction_after_init}\n")


    # Step 2: Manually mutate the instruction inside the module.
    new_instruction = "THIS IS A NEW MUTATED INSTRUCTION TO TEST BUG."
    print(f"2. Manually updating instruction to: '{new_instruction}'")
    predictor.predict.signature.instructions = new_instruction
    print(f"   Update complete.\n")
    

    # Step 3: Call dump_state() to see if it reflects the change.
    print("3. Calling dump_state() to inspect the saved state...")
    dumped_state = predictor.dump_state()

    # Step 4: Analyze the result.
    if 'signature' in dumped_state and 'instructions' in dumped_state['signature']:
        instruction_from_dump_state = dumped_state['signature']['instructions']
        print(f"   -> Instruction from dump_state(): '{instruction_from_dump_state}'")
        
        if instruction_from_dump_state == new_instruction:
            print("\n✅ SUCCESS: dump_state() correctly reflects the mutated instruction. No bug detected here.")
            return True
        else:
            print("\n❌ FAILURE: dump_state() shows the OLD instruction. This indicates a bug in dspy.ChainOfThought.")
            print(f"   Expected: '{new_instruction}'")
            print(f"   Got:      '{instruction_from_dump_state}'")
            return False
    else:
        print("\n❌ ERROR: Could not find 'instructions' key in the dumped state.")
        print("   Dumped state keys:", list(dumped_state.keys()))
        return False

if __name__ == "__main__":
    # Configure a dummy LM to prevent errors
    with dspy.context(lm=dspy.LM(model="dummy_model", api_key="dummy_key")):
        test_chain_of_thought_state_bug()
