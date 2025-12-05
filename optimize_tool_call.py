# --- 1. Imports and Setup ---
import dspy
import os
from datetime import datetime
from pydantic import BaseModel
import harness

#GLM_46='openai/glm-4.6',
GLM_46='openai/z-ai/glm-4.6'

# --- Setup LM ---
# Configure the language model and adapter.
zai_glm_4_6 = dspy.LM(
    model=GLM_46,
    api_key=os.getenv("ZAI_API_KEY", "your-api-key-here"),
    api_base=os.getenv("ZAI_API_BASE", "your-api-base-here"),
    cache=False,
    disable_reasoning=True,
)

# Use chat adapter to simulate real world conditions.
dspy.configure(lm=zai_glm_4_6, adapter=dspy.ChatAdapter(), experimental=True)


# --- 2. Program Definition ---
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


class ToolSignature(dspy.Signature):
    __doc__ = final_system_message

    query = dspy.InputField(desc="User's query asking for a tool call.")
    tool_call: harness.ToolCall = dspy.OutputField(desc="A valid JSON object representing a tool call.")


class ToolCaller(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(ToolSignature)

    def forward(self, query):
        # Simulate a long chat history to make the task harder.
        long_conversation_text = '''User: Hello, can you help me plan a trip?
Assistant: Of course! I can help with that. Where would you like to go and when?
User: I was thinking of a trip to Europe in the fall. Maybe September or October.
Assistant: That's a wonderful time to visit Europe. The weather is pleasant and the crowds are smaller. Do you have any specific countries or cities in mind?
User: I've always wanted to see Italy. Rome, Florence, and Venice are on my list.
Assistant: Excellent choices. Italy is beautiful in the fall. A classic itinerary would be a few days in each of those cities. They are all well-connected by train.
User: That sounds great. What about accommodation? I prefer boutique hotels.
Assistant: I can certainly look up some highly-rated boutique hotels for you in each city. Do you have a budget in mind per night?
User: Let's say around 200-300 Euros per night.
Assistant: Understood. I will find some options for you within that price range. Besides sightseeing, are there any specific activities you're interested in, like cooking classes or wine tasting?
User: A cooking class in Florence sounds amazing! And maybe a gondola ride in Venice.
Assistant: Noted. A cooking class in Tuscany and a gondola ride are classic experiences. I'll add them to the plan. Is there anything else I can help you with for this trip?
User: What about flights? I'll be flying from New York.
Assistant: I can search for flight options for you. Which airport in New York would you be flying from? JFK, LaGuardia, or Newark?
User: JFK would be best.
Assistant: Alright, I will look for round-trip flights from JFK to Rome and back from Venice. I'll compile all this information for you.
User: Thanks! Also, I need to know about something else completely unrelated.
Assistant: I'm here to help. What is it?
'''
        # Repeat the text to simulate ~3000 tokens (1 token ~= 4 chars)
        simulated_history = long_conversation_text * 7
        full_query = simulated_history + query

        prediction = self.predictor(query=full_query)
        return prediction

trainset = [x.with_inputs('query') for x in harness.train_data]
devset = [x.with_inputs('query') for x in harness.dev_data]


# --- 3. Optimizer ---
from dspy.teleprompt import GEPA

# GEPA is a reflective optimizer that can use textual feedback from metrics
# to generate improved prompts.
optimizer = GEPA(
    metric=harness.validation_metric_with_feedback,
    auto="light",
    track_stats=True,
    reflection_minibatch_size=1,
    # A powerful model is recommended for the reflection step.
    # You can remove `reflection_lm` if you don't have access to a different model.
    reflection_lm=dspy.LM(
        model=GLM_46,
        api_key=os.getenv("ZAI_API_KEY", "your-api-key-here"),
        api_base=os.getenv("ZAI_API_BASE", "your-api-base-here"),
        #cache=False,
        #disable_reasoning=True,
    )
)


# --- 4. Execution ---
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

    print("\n--- Evaluating Unoptimized Program ---")
    from dspy.evaluate import Evaluate

    try:
        import pandas as pd
        display_table = 5
    except ImportError:
        print("`pandas` not installed. Skipping table display.")
        display_table = False

    evaluate = Evaluate(devset=devset, metric=harness.validation_metric_with_feedback, num_threads=1, display_progress=True, display_table=display_table)
    eval_result = evaluate(program_to_optimize)
    print(f"\nInitial score on dev set: {eval_result.score if eval_result else 'N/A'}")


    # Compile the program to find an optimized prompt.
    print("\n--- Compiling Program ---")
    optimized_program = optimizer.compile(
        program_to_optimize,
        trainset=trainset,
        valset=devset,
    )

    # View the optimized prompt.
    optimized_program(query="weather in berlin")
    print("\n--- Optimized Program's Prompt ---")
    if len(zai_glm_4_6.history) > 1:
        final_prompt = zai_glm_4_6.history[-1]['messages'][-1]['content']
        print(final_prompt)

        # Save prompt to a timestamped file to preserve each run's output.
        PROMPTS_DIR = "prompts"
        os.makedirs(PROMPTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(PROMPTS_DIR, f"optimized_tool_call_{timestamp}.txt")

        with open(file_path, "w") as f:
            f.write(final_prompt)
        print(f"\nFinal prompt saved to '{file_path}'")

    print("\n--- Evaluating Optimized Program ---")
    evaluate = Evaluate(devset=devset, metric=harness.validation_metric_with_feedback, num_threads=1, display_progress=True, display_table=display_table)
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

        harness.ToolCall.model_validate_json(tool_call_output)
        print("\nLive test output is VALID.")
    except Exception as e:
        print(f"\nLive test output is INVALID: {e}")
