# --- 1. Imports and Setup ---
import dspy
import os
import json
from pydantic import BaseModel, Field, ValidationError

# --- Setup LM ---
# Configure the language model and adapter.
# The user mentioned 'zai-glm-4.6', which we will treat as an OpenAI-compatible model endpoint.
zai_glm_4_6 = dspy.LM(
    model='openai/zai-glm-4.6',
    api_key=os.getenv("ZAI_API_KEY", "your-api-key-here"),
    api_base=os.getenv("ZAI_API_BASE", "your-api-base-here"),
    cache=False,
)

# Use JSONAdapter to guide the model towards valid JSON output.
# This adapter leverages a model's native JSON mode or function-calling capabilities if available.
dspy.configure(lm=zai_glm_4_6, adapter=dspy.JSONAdapter(), experimental=True)


# --- 2. Schema Definition (Pydantic) ---
# Define the expected JSON structure using Pydantic models.
# This provides clear type hints (e.g., float) that DSPy uses to guide the LM.
class GetWeatherInput(BaseModel):
    """Input schema for the get_weather tool."""
    longitude: float
    latitude: float

class ToolCall(BaseModel):
    """The general tool-calling JSON object."""
    tool_name: str = Field(..., pattern="^get_weather$")
    tool_input: GetWeatherInput


# --- 3. Program Definition ---
# The initial prompt, including the JSON schema, is placed in the signature's docstring.
class ToolSignature(dspy.Signature):
    """You have access to the following tools.
To use a tool, you MUST respond with a single JSON object exclusively in the following format (do not include ```json ... ``` or any other text):
{
  "tool_name": "NAME_OF_THE_TOOL",
  "tool_input": { /* parameters for the tool as a JSON object, matching the schema provided for the tool */ }
}

Do NOT include any other text, explanation, or conversational filler before or after the JSON object if you are calling a tool.
If you are not calling a tool, respond to the user as a helpful assistant.

Available tools:
<tools>
[
  {
    "name": "get_weather",
    "description": "Get the weather for a given location",
    "parameters": {"$schema":"http://json-schema.org/draft-07/schema#","required":["latitude","longitude"],"properties":{"longitude":{"type":"number"},"latitude":{"type":"number"}},"additionalProperties":false,"type":"object"}
  }
]
</tools>
"""
    query = dspy.InputField(desc="User's query asking for a tool call.")
    tool_call: ToolCall = dspy.OutputField(desc="A valid JSON object representing a tool call.")


class ToolCaller(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(ToolSignature)

    def forward(self, query):
        prediction = self.predictor(query=query)
        return prediction


# --- 4. Data ---
# The "gold" standard for this task is a valid JSON string that conforms to the schema.
train_data = [
    dspy.Example(
        query="what's the weather like in san francisco?",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=37.7749, longitude=-122.4194)).model_dump_json()
    ).with_inputs("query"),
    dspy.Example(
        query="tell me the weather for tokyo",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=35.6895, longitude=139.6917)).model_dump_json()
    ).with_inputs("query"),
]

dev_data = [
    dspy.Example(
        query="weather in london please",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=51.5072, longitude=-0.1276)).model_dump_json()
    ).with_inputs("query"),
    dspy.Example(
        query="how is the weather in new york city",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=40.7128, longitude=-74.0060)).model_dump_json()
    ).with_inputs("query"),
    # Add more varied examples to better test the model
    dspy.Example(
        query="I'm in Sydney, what's the weather like?",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=-33.8688, longitude=151.2093)).model_dump_json()
    ).with_inputs("query"),
    dspy.Example(
        query="give me the forecast for Paris",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=48.8566, longitude=2.3522)).model_dump_json()
    ).with_inputs("query"),
    dspy.Example(
        query="is it sunny in cairo",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=30.0444, longitude=31.2357)).model_dump_json()
    ).with_inputs("query"),
    dspy.Example(
        query="What's the weather in Moscow?",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=55.7558, longitude=37.6173)).model_dump_json()
    ).with_inputs("query"),
    dspy.Example(
        query="Tell me about the weather in Rio de Janeiro",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=-22.9068, longitude=-43.1729)).model_dump_json()
    ).with_inputs("query"),
    dspy.Example(
        query="How's Beijing's weather?",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=39.9042, longitude=116.4074)).model_dump_json()
    ).with_inputs("query"),
    dspy.Example(
        query="I need the weather for Cape Town",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=-33.9249, longitude=18.4241)).model_dump_json()
    ).with_inputs("query"),
    dspy.Example(
        query="Mumbai weather forecast",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=19.0760, longitude=72.8777)).model_dump_json()
    ).with_inputs("query"),
]

trainset = [x.with_inputs('query') for x in train_data]
devset = [x.with_inputs('query') for x in dev_data]


# --- 5. Metric Definition ---
def validation_metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Validates the predicted tool_call JSON against the Pydantic schema.
    Returns a score of 1 if valid, 0 otherwise, and provides specific
    feedback on validation errors for the GEPA optimizer.
    """
    raw_output = pred.tool_call
    if isinstance(raw_output, BaseModel):
        raw_output = raw_output.model_dump_json()

    try:
        ToolCall.model_validate_json(raw_output)
        return dspy.Prediction(score=1.0, feedback="Correct format and types.")
    except ValidationError as e:
        feedback = f"JSON validation failed. Ensure all field types are correct (e.g., numbers are not strings). Error: {e}"
        return dspy.Prediction(score=0.0, feedback=feedback)
    except (json.JSONDecodeError, TypeError, Exception) as e:
        feedback = f"Invalid JSON or unexpected error. Raw output: '{raw_output}'. Error: {e}"
        return dspy.Prediction(score=0.0, feedback=feedback)


# --- 6. Optimizer ---
from dspy.teleprompt import GEPA

# GEPA is a reflective optimizer that can use textual feedback from metrics
# to generate improved prompts.
optimizer = GEPA(
    metric=validation_metric_with_feedback,
    auto="light",
    track_stats=True,
    reflection_minibatch_size=1,
    # A powerful model is recommended for the reflection step.
    # You can remove `reflection_lm` if you don't have access to a different model.
    reflection_lm=dspy.LM(
        model='openai/gpt-4o',
        api_key=os.getenv("OPENAI_API_KEY", "your-openai-key-here"),
    )
)


# --- 7. Execution ---
if __name__ == "__main__":
    # Add a direct test call to diagnose connection issues.
    print("--- Running a direct API call to test configuration ---")
    try:
        response = zai_glm_4_6("This is a test. Respond with OK.")
        print(f"--- Direct API call successful. Response: {response} ---")
    except Exception as e:
        print(f"\n--- !!! Direct API call FAILED. This is why you see no API usage. !!! ---")
        print(f"Error: {e}")
        print("--- Please check your ZAI_API_KEY, ZAI_API_BASE, and network connection. ---")
        exit()

    print("\n--- Starting Tool Call Optimization ---")

    program_to_optimize = ToolCaller()

    # View the unoptimized prompt by running it with a dummy input.
    program_to_optimize(query="weather in paris")
    print("\n--- Unoptimized Program's Prompt ---")
    if zai_glm_4_6.history:
        print(zai_glm_4_6.history[-1]['messages'][-1]['content'])

    # Compile the program to find an optimized prompt.
    optimized_program = optimizer.compile(
        program_to_optimize,
        trainset=trainset,
        valset=devset,
    )

    # View the optimized prompt.
    optimized_program(query="weather in berlin")
    print("\n--- Optimized Program's Prompt ---")
    if len(zai_glm_4_6.history) > 1:
        print(zai_glm_4_6.history[-1]['messages'][-1]['content'])

    print("\n--- Evaluating Optimized Program ---")
    from dspy.evaluate import Evaluate

    try:
        import pandas as pd
        display_table = 5
    except ImportError:
        print("`pandas` not installed. Skipping table display.")
        display_table = False

    evaluate = Evaluate(devset=devset, metric=validation_metric_with_feedback, num_threads=1, display_progress=True, display_table=display_table)
    eval_result = evaluate(optimized_program)
    print(f"\nFinal score on dev set: {eval_result.score if eval_result else 'N/A'}")

    # Test with a live example to see the final output.
    print("\n--- Live Test ---")
    live_test = optimized_program(query="weather in cairo")
    print(f"Query: weather in cairo")
    print(f"Output:\n{live_test.tool_call}")

    # Manually validate the live test output to confirm correctness.
    try:
        tool_call_output = live_test.tool_call
        if isinstance(tool_call_output, BaseModel):
            tool_call_output = tool_call_output.model_dump_json()

        ToolCall.model_validate_json(tool_call_output)
        print("\nLive test output is VALID.")
    except Exception as e:
        print(f"\nLive test output is INVALID: {e}")
