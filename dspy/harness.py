import dspy
import json
from pydantic import BaseModel, ValidationError

def create_validation_metric(tool_schema: type[BaseModel]):
    """
    Returns a metric function that validates a predicted tool_call JSON
    against the provided Pydantic schema.
    """
    def validation_metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
        raw_output = pred.tool_call
        if isinstance(raw_output, BaseModel):
            raw_output = raw_output.model_dump_json()

        try:
            tool_schema.model_validate_json(raw_output)
            return dspy.Prediction(score=1.0, feedback="Correct format and types.")
        except ValidationError as e:
            feedback = f"JSON validation failed. Ensure all field types are correct (e.g., numbers are not strings). Error: {e}"
            return dspy.Prediction(score=0.0, feedback=feedback)
        except (json.JSONDecodeError, TypeError, Exception) as e:
            feedback = f"Invalid JSON or unexpected error. Raw output: '{raw_output}'. Error: {e}"
            return dspy.Prediction(score=0.0, feedback=feedback)

    return validation_metric_with_feedback
