import dspy
import os

# --- 1. Setup ---
# This section configures the language model.
# Replace the placeholders with your actual model details and credentials.
# The user mentioned 'zai-glm-4.6', which we will treat as an OpenAI-compatible model endpoint.
zai_glm_4_6 = dspy.LM(
    model='openai/zai-glm-4.6',
    api_key=os.getenv("ZAI_API_KEY", "your-api-key-here"),
    api_base=os.getenv("ZAI_API_BASE", "your-api-base-here"),
    # Add any other necessary parameters for your model here.
)

# Set the configured language model as the default for all dspy modules.
dspy.configure(lm=zai_glm_4_6)


# --- 2. Data ---
# This section defines the training and development datasets.
# In a real-world scenario, you would load this data from files (e.g., CSV, JSON).
train_data = [
    dspy.Example(text="dspy is a powerful framework", sentiment="Positive").with_inputs("text"),
    dspy.Example(text="I'm not sure if I like this", sentiment="Neutral").with_inputs("text"),
    dspy.Example(text="This is the worst thing ever", sentiment="Negative").with_inputs("text"),
    dspy.Example(text="I love how easy it is to use", sentiment="Positive").with_inputs("text"),
]
dev_data = [
    dspy.Example(text="This is fantastic!", sentiment="Positive").with_inputs("text"),
    dspy.Example(text="I have no strong feelings about this.", sentiment="Neutral").with_inputs("text"),
    dspy.Example(text="It's a complete disaster.", sentiment="Negative").with_inputs("text"),
]

trainset = [x.with_inputs('text') for x in train_data]
devset = [x.with_inputs('text') for x in dev_data]


# --- 3. Program Definition ---
# This section defines the DSPy program, which is the prompt template to be optimized.

# A dspy.Signature defines the input and output fields of a module.
# It's a declarative way to specify what a module should do.
class SentimentSignature(dspy.Signature):
    """Classify sentiment as Positive, Negative, or Neutral."""

    # Sample for a natural language constraint:
    # By describing the expected format in the OutputField's description,
    # we guide the language model.
    text = dspy.InputField(desc="The text to classify.")
    sentiment = dspy.OutputField(desc="One of: Positive, Negative, or Neutral.")


# A dspy.Module is a building block of a DSPy program.
class SentimentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        # dspy.Predict uses the signature to generate a prompt and call the LM.
        self.predictor = dspy.Predict(SentimentSignature)

    def forward(self, text):
        prediction = self.predictor(text=text)

        return prediction


# --- 4. Metric Definition ---
# This section defines a metric to evaluate the program's performance.
# The optimizer will try to maximize this metric's score.
def sentiment_metric(gold, pred, trace=None):
    """A simple metric that checks for exact match of sentiment."""
    return gold.sentiment.lower() == pred.sentiment.lower()


# --- 5. Optimizer ---
# This section configures the prompt optimizer (teleprompter).
# We'll use BootstrapFewShot, which is effective for generating prompts with demonstrations.
from dspy.teleprompt import BootstrapFewShot

# Optimizer configuration
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

# Instantiate the optimizer.
optimizer = BootstrapFewShot(metric=sentiment_metric, **config)


# --- 6. Execution ---
if __name__ == "__main__":
    print("--- Starting Prompt Optimization ---")

    # Instantiate the program we want to optimize.
    unoptimized_program = SentimentClassifier()

    # Compile the program. The optimizer will find the best prompt and demonstrations.
    optimized_program = optimizer.compile(unoptimized_program, trainset=trainset)

    print("\n--- Optimized Program's Prompt ---")
    # Inspect the optimized prompt and demonstrations.
    optimized_program.predictor.dump_state()

    print("\n--- Evaluating Optimized Program ---")
    # Evaluate the performance of the optimized program on the development set.
    from dspy.evaluate import Evaluate

    evaluate = Evaluate(devset=devset, metric=sentiment_metric, num_threads=1, display_progress=True, display_table=5)
    eval_result = evaluate(optimized_program)
    print(f"\nFinal score on dev set: {eval_result.score}")
