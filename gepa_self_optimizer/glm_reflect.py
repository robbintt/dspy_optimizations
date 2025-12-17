import dspy
from gepa_config import JUDGE_CONSTITUTION

class Generate(dspy.Signature):
    """Generate a comprehensive answer to a given question, using step-by-step reasoning."""
    question = dspy.InputField(desc="The question to be answered.")
    draft_answer = dspy.OutputField(desc="A comprehensive, step-by-step answer to the question.")

class Critic(dspy.Signature):
    """Act as a ruthless critic. Analyze the draft for errors based on the provided constitution."""
    constitution = dspy.InputField(desc="The principles for judging the draft.")
    question = dspy.InputField(desc="The question the draft is trying to answer.")
    draft_answer = dspy.InputField(desc="The draft answer to be critiqued.")
    critique = dspy.OutputField(desc="A list of specific errors found in the draft.")
    severity = dspy.OutputField(desc="The severity of the errors: High, Medium, or Low.")

class Refine(dspy.Signature):
    """Rewrite the draft to fix all errors identified in the critique to produce a final, correct answer."""
    question = dspy.InputField(desc="The original question.")
    draft_answer = dspy.InputField(desc="The draft answer that contains errors.")
    critique = dspy.InputField(desc="The list of errors to be fixed.")
    final_answer = dspy.OutputField(desc="The refined and correct final answer.")

class GlmReflect(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(Generate)
        self.critic = dspy.Predict(Critic)
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
            return dspy.Prediction(answer=draft_answer, critique=critique_pkg.critique, severity=critique_pkg.severity)
