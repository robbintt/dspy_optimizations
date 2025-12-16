import dspy
from gepa_config import JUDGE_CONSTITUTION

# --- SIGNATURES ---
class Generate(dspy.Signature):
    """Generate a comprehensive answer. Use System 2 thinking."""
    question = dspy.InputField()
    draft_answer = dspy.OutputField()

class ShepherdCritic(dspy.Signature):
    """Act as a ruthless critic. Analyze the draft for errors based on the Constitution."""
    constitution = dspy.InputField()
    question = dspy.InputField()
    draft_answer = dspy.InputField()
    critique = dspy.OutputField(desc="List of specific errors")
    severity = dspy.OutputField(desc="High, Medium, or Low")

class Refine(dspy.Signature):
    """Rewrite the draft to fix the errors identified in the critique."""
    question = dspy.InputField()
    draft_answer = dspy.InputField()
    critique = dspy.InputField()
    final_answer = dspy.OutputField()

# --- THE MODULE ---
class GlmSelfReflect(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(Generate)
        self.critic = dspy.ChainOfThought(ShepherdCritic)
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
