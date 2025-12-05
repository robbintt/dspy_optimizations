import dspy
import json
from pydantic import BaseModel, Field, ValidationError

# --- Schema Definition (Pydantic) ---
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


# --- Data ---
# The "gold" standard for this task is a valid JSON string that conforms to the schema.
train_data = [
    dspy.Example(
        query="what's the weather like in san francisco?",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=37.7749, longitude=-122.4194)).model_dump_json()
    ),
    dspy.Example(
        query="tell me the weather for tokyo",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=35.6895, longitude=139.6917)).model_dump_json()
    ),
]

dev_data = [
    dspy.Example(
        query="weather in london please",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=51.5072, longitude=-0.1276)).model_dump_json()
    ),
    dspy.Example(
        query="how is the weather in new york city",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=40.7128, longitude=-74.0060)).model_dump_json()
    ),
    # Add more varied examples to better test the model
    dspy.Example(
        query="I'm in Sydney, what's the weather like?",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=-33.8688, longitude=151.2093)).model_dump_json()
    ),
    dspy.Example(
        query="give me the forecast for Paris",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=48.8566, longitude=2.3522)).model_dump_json()
    ),
    dspy.Example(
        query="is it sunny in cairo",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=30.0444, longitude=31.2357)).model_dump_json()
    ),
    dspy.Example(
        query="What's the weather in Moscow?",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=55.7558, longitude=37.6173)).model_dump_json()
    ),
    dspy.Example(
        query="Tell me about the weather in Rio de Janeiro",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=-22.9068, longitude=-43.1729)).model_dump_json()
    ),
    dspy.Example(
        query="How's Beijing's weather?",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=39.9042, longitude=116.4074)).model_dump_json()
    ),
    dspy.Example(
        query="I need the weather for Cape Town",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=-33.9249, longitude=18.4241)).model_dump_json()
    ),
    dspy.Example(
        query="Mumbai weather forecast",
        tool_call=ToolCall(tool_name="get_weather", tool_input=GetWeatherInput(latitude=19.0760, longitude=72.8777)).model_dump_json()
    ),
]


# --- System Prompt ---
final_system_message = """You are a helpful AI assistant. 
***
Description:
Assistant is an AI assistant to memgrafter.
***
Persona:
I am a 40 year old programmer. I am interested in text games, science fiction, and programming. I like to build code projects, woodworking and other fabrication, and read speculative nonfiction or harder science fiction.

I am 6'2" and 205 lbs. My workouts are 90 minute zone 2 runs and olympic barbell.


***

You have access to the following tools.
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
    "parameters": {"$schema":"http:\/\/json-schema.org\/draft-07\/schema#","required":["latitude","longitude"],"properties":{"longitude":{"type":"number"},"latitude":{"type":"number"}},"additionalProperties":false,"type":"object"}
  }
]
</tools>"""


# --- Metric Definition ---
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
