# DSPy v3.0.4 Reference Guide

> Comprehensive digest of key functions, classes, and interfaces for DSPy version 3.0.4

**Package Information:**
- Version: 3.0.4b2
- Author: Omar Khattab
- Repository: https://github.com/stanfordnlp/dspy

---

## Table of Contents

1. [Package Structure](#package-structure)
2. [Core Abstractions](#core-abstractions)
3. [Prediction Modules](#prediction-modules)
4. [Language Model Clients](#language-model-clients)
5. [Adapters](#adapters)
6. [Optimizers (Teleprompt)](#optimizers-teleprompt)
7. [Evaluation](#evaluation)
8. [Retrievers](#retrievers)
9. [Configuration & Settings](#configuration--settings)
10. [Datasets](#datasets)
11. [Utilities](#utilities)
12. [Public API Summary](#public-api-summary)
13. [Key Patterns & Workflows](#key-patterns--workflows)

---

## Package Structure

### Main Directories

```
dspy/
├── primitives/        # Core abstractions (Module, Example, Prediction)
├── signatures/        # Signature system for I/O definitions
├── predict/          # Prediction modules (Predict, ChainOfThought, ReAct, etc.)
├── clients/          # Language model clients and interfaces
├── adapters/         # Format adapters (Chat, JSON, XML)
├── teleprompt/       # Optimizers/teleprompters
├── evaluate/         # Evaluation utilities
├── retrievers/       # Retrieval modules
├── datasets/         # Dataset loaders
├── streaming/        # Streaming support
├── utils/           # Utility functions
└── experimental/    # Experimental features
```

---

## Core Abstractions

### Module

**Location:** `dspy/primitives/module.py`

Base class for all DSPy programs. All DSPy modules inherit from this class.

**Key Methods:**

```python
class Module:
    def __call__(*args, **kwargs)
        # Executes the module (calls forward)

    def forward(*args, **kwargs)
        # Main execution method (must be implemented by subclasses)

    async def acall()
    async def aforward()
        # Async versions of call/forward

    def named_predictors()
        # Get all Predict modules in this module

    def batch(examples, num_threads=None)
        # Process examples in parallel

    def save(path, save_program=False)
    @staticmethod
    def load(path)
        # Serialization methods

    def deepcopy()
    def reset_copy()
        # Copying utilities

    def inspect_history(n=1)
        # View LM call history
```

### Example

**Location:** `dspy/primitives/example.py`

Data container for training/evaluation with dictionary-like interface.

**Key Methods:**

```python
class Example:
    def with_inputs(*keys)
        # Mark specified fields as inputs

    def inputs()
        # Get input fields

    def labels()
        # Get output/label fields

    def copy(**kwargs)
        # Create a copy with optional overrides

    def without(*keys)
        # Create copy without specified fields

    def toDict()
        # Serialize to dictionary
```

**Usage Example:**

```python
example = dspy.Example(
    question="What is 2+2?",
    answer="4"
).with_inputs("question")
```

### Prediction

**Location:** `dspy/primitives/prediction.py`

Extends Example for module outputs. Contains completions and usage information.

**Key Attributes:**

```python
class Prediction(Example):
    _completions      # Multiple completions (if n>1)
    _lm_usage        # Token usage information
```

Supports arithmetic/comparison operations on the `score` field.

### Signature

**Location:** `dspy/signatures/signature.py`

Defines input/output schema for modules. Built on Pydantic BaseModel.

**Creation Methods:**

```python
# Class-based
class QA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")

# String-based
signature = dspy.Signature("question -> answer")

# Programmatic
signature = dspy.make_signature(
    "question -> answer",
    instructions="Answer questions with short factoid answers."
)

# Normalize to Signature
sig = dspy.ensure_signature(sig_or_string)
```

**Manipulation Methods:**

```python
class Signature:
    def prepend(name, field, type_=None)
        # Add field at beginning

    def append(name, field, type_=None)
        # Add field at end

    def insert(index, name, field, type_=None)
        # Insert field at specific position

    def delete(name)
        # Remove field

    def with_instructions(instructions)
        # Update signature instructions

    def with_updated_fields(name, **kwargs)
        # Update field properties
```

### Field Types

**Location:** `dspy/signatures/field.py`

```python
# Input field
question: str = dspy.InputField(
    desc="The question to answer",
    prefix="Question:",
    format=lambda x: x.strip()
)

# Output field
answer: str = dspy.OutputField(
    desc="Short factoid answer",
    prefix="Answer:"
)
```

---

## Prediction Modules

**Location:** `dspy/predict/`

### Predict

Basic LM call with signature.

```python
predict = dspy.Predict(signature, **config)
result = predict(question="What is 2+2?")
```

**Config options:** `temperature`, `max_tokens`, `n`, `stop`, etc.

### ChainOfThought

Adds reasoning step before outputs.

```python
cot = dspy.ChainOfThought(
    signature,
    rationale_field=None,  # Custom reasoning field name
    **config
)
result = cot(question="What is 2+2?")
print(result.rationale)  # The reasoning
print(result.answer)     # The answer
```

### ReAct

Reasoning + Acting with tools.

```python
react = dspy.ReAct(
    signature,
    tools=[search_tool, calculator_tool],
    max_iters=10
)
result = react(question="What is the population of Paris?")
```

### ProgramOfThought

Generates and executes Python code.

```python
pot = dspy.ProgramOfThought(
    signature,
    max_iters=3,
    interpreter=None  # Optional custom interpreter
)
result = pot(question="Calculate 15% of 250")
```

**Note:** Requires Deno installed for code execution.

### Refine

Runs module multiple times, returns best result.

```python
refine = dspy.Refine(
    module=base_module,
    N=5,                    # Number of attempts
    reward_fn=metric,       # Scoring function
    threshold=0.9,          # Stop if reached
    fail_count=None        # Max failures allowed
)
```

### BestOfN

Generate N outputs, select best by metric.

```python
best_of_n = dspy.BestOfN(
    module=base_module,
    N=5,
    metric=my_metric
)
```

### MultiChainComparison

Compare multiple reasoning chains.

```python
mcc = dspy.MultiChainComparison(
    signature,
    M=3,  # Number of reasoning chains
    **config
)
```

### Parallel

Execute multiple examples in parallel.

```python
parallel = dspy.Parallel(
    num_threads=4,
    max_errors=0
)
results = parallel(module, examples=[ex1, ex2, ex3])
```

### KNN

K-nearest neighbor retrieval + prediction.

```python
knn = dspy.KNN(
    k=3,
    trainset=examples
)
```

### CodeAct

Code generation and execution agent.

```python
codeact = dspy.CodeAct(
    signature,
    tools=[...],
    max_iters=10
)
```

---

## Language Model Clients

**Location:** `dspy/clients/`

### LM

Main language model client using LiteLLM.

```python
lm = dspy.LM(
    model="openai/gpt-4o-mini",
    model_type="chat",         # or "text"
    temperature=None,
    max_tokens=None,
    cache=True,
    api_key=None,
    **kwargs                   # Provider-specific kwargs
)
```

**Supported Providers:** OpenAI, Anthropic, Google, Cohere, Azure, etc.

**Methods:**

```python
# Sync call
response = lm.forward(
    prompt=None,
    messages=None,
    **kwargs
)

# Async call
response = await lm.aforward(
    prompt=None,
    messages=None,
    **kwargs
)

# Direct call (with callbacks)
response = lm(prompt="Hello", temperature=0.7)
```

### BaseLM

Base class for custom LM implementations.

```python
class CustomLM(dspy.BaseLM):
    def forward(self, prompt=None, messages=None, **kwargs):
        # Must return OpenAI-compatible response format
        return {
            "choices": [{
                "message": {
                    "content": "response text",
                    "role": "assistant"
                }
            }],
            "usage": {...}
        }
```

### Embedder

For generating embeddings.

```python
embedder = dspy.Embedder(
    model="openai/text-embedding-3-small",
    **kwargs
)
embeddings = embedder(["text1", "text2"])
```

### Provider

Provider abstraction for fine-tuning/training.

```python
class Provider:
    def TrainingJob(...)
        # Create fine-tuning job

    def ReinforceJob(...)
        # Create reinforcement learning job
```

---

## Adapters

**Location:** `dspy/adapters/`

Adapters control how signatures are formatted for LM calls.

### ChatAdapter

Default adapter for chat-based models.

```python
adapter = dspy.ChatAdapter()
```

Formats fields with headers: `[[ ## field_name ## ]]`

### JSONAdapter

Forces JSON output format.

```python
adapter = dspy.JSONAdapter()
```

Fallback when ChatAdapter fails or for structured output.

### XMLAdapter

XML-based formatting.

```python
adapter = dspy.XMLAdapter()
```

### TwoStepAdapter

Two-phase formatting approach.

```python
adapter = dspy.TwoStepAdapter()
```

### Adapter Configuration

```python
adapter = dspy.ChatAdapter(
    use_native_function_calling=True,
    native_response_types=[dspy.Citations, dspy.Document]
)
```

### Custom Types

**Location:** `dspy/adapters/types/`

```python
# Image input
image = dspy.Image(url="https://...", detail="high")

# Audio input
audio = dspy.Audio(url="https://...")

# Conversation history
history = dspy.History([...])

# Tool definition
tool = dspy.Tool(
    name="search",
    desc="Search the web",
    func=search_function
)

# Code blocks
code = dspy.Code(code="print('hello')", language="python")
```

---

## Optimizers (Teleprompt)

**Location:** `dspy/teleprompt/`

### BootstrapFewShot

Generate few-shot examples from training set.

```python
optimizer = dspy.BootstrapFewShot(
    metric=my_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    max_rounds=1,
    max_errors=5,
    teacher_settings={}
)

optimized = optimizer.compile(
    student=MyModule(),
    teacher=None,  # Optional teacher module
    trainset=train_examples
)
```

### BootstrapFewShotWithRandomSearch

Bootstrap + random search over demos.

```python
optimizer = dspy.BootstrapRS(
    metric=my_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    num_candidate_programs=10,
    num_threads=4
)

optimized = optimizer.compile(
    student=MyModule(),
    trainset=train_examples,
    valset=val_examples
)
```

### MIPROv2

Multi-prompt Instruction Proposal Optimizer v2.

```python
optimizer = dspy.MIPROv2(
    metric=my_metric,
    auto="medium",  # "light", "medium", or "heavy"
    num_candidates=10,
    init_temperature=1.0,
    verbose=False
)

optimized = optimizer.compile(
    student=MyModule(),
    trainset=train_examples,
    valset=val_examples,
    num_trials=100,
    max_bootstrapped_demos=4,
    max_labeled_demos=16
)
```

### COPRO

Coordinate Prompt Optimization.

```python
optimizer = dspy.COPRO(
    metric=my_metric,
    depth=3,
    breadth=10
)

optimized = optimizer.compile(
    student=MyModule(),
    trainset=train_examples,
    eval_kwargs={"num_threads": 4}
)
```

### LabeledFewShot

Simple few-shot with labeled examples.

```python
optimizer = dspy.LabeledFewShot(k=3)
optimized = optimizer.compile(
    student=MyModule(),
    trainset=train_examples
)
```

### BootstrapFinetune

Bootstrap for fine-tuning.

```python
optimizer = dspy.BootstrapFinetune(
    metric=my_metric,
    num_epochs=3,
    learning_rate=1e-5
)

optimized = optimizer.compile(
    student=MyModule(),
    trainset=train_examples
)
```

### Other Optimizers

- **Ensemble** - Ensemble multiple programs
- **KNNFewShot** - KNN-based demo selection
- **AvatarOptimizer** - Avatar-based optimization
- **GEPA** - Generalized Automatic Prompt Engineering
- **SIMBA** - Simulation-based optimization
- **InferRules** - Rule inference optimizer

---

## Evaluation

**Location:** `dspy/evaluate/`

### Evaluate

Main evaluation class.

```python
evaluator = dspy.Evaluate(
    devset=dev_examples,
    metric=my_metric,
    num_threads=None,
    display_progress=False,
    display_table=False,
    max_errors=None,
    return_all_scores=False,
    return_outputs=False
)

# Run evaluation
result = evaluator(program)
# Returns: EvaluationResult(score=0.85, results=[...])
```

### Built-in Metrics

```python
# Exact match
def my_metric(example, pred, trace=None):
    return dspy.answer_exact_match(example, pred)

# Or use the EM metric directly
evaluator = dspy.Evaluate(devset=dev, metric=dspy.EM)

# Passage match
dspy.answer_passage_match(example, pred, trace=None)

# Text normalization
normalized = dspy.normalize_text(text)
```

### Auto Evaluation Metrics

```python
# Semantic F1
metric = dspy.SemanticF1()

# Complete and grounded
metric = dspy.CompleteAndGrounded()
```

### Custom Metrics

```python
def custom_metric(example, pred, trace=None):
    """
    Args:
        example: The gold example with inputs/labels
        pred: The prediction from the model
        trace: Optional trace information

    Returns:
        float or bool: Score (higher is better)
    """
    return pred.answer.lower() == example.answer.lower()
```

---

## Retrievers

**Location:** `dspy/retrievers/`

### Retrieve

Base retrieval module.

```python
# Configure retrieval model
dspy.configure(rm=my_retrieval_model)

# Use retriever
retrieve = dspy.Retrieve(k=3)
result = retrieve(query="What is DSPy?")
# Returns: Prediction(passages=[...])
```

**Passage Format:**

```python
{
    "long_text": "The full passage text...",
    "pid": 123,  # Optional passage ID
    "score": 0.95  # Optional relevance score
}
```

### Embeddings

Embedding-based retrieval using FAISS.

```python
embedder = dspy.Embedder("openai/text-embedding-3-small")

retriever = dspy.Embeddings(
    corpus=documents,
    embedder=embedder,
    k=5,
    index_type="auto"  # "auto", "simple", or "faiss"
)

result = retriever(query="What is DSPy?")
```

---

## Configuration & Settings

### Global Configuration

```python
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    rm=my_retriever,
    adapter=dspy.ChatAdapter(),
    callbacks=[],
    track_usage=False,
    async_max_workers=8,
    experimental=False
)
```

### Settings Access

```python
# Access current settings
current_lm = dspy.settings.lm
current_rm = dspy.settings.rm
current_adapter = dspy.settings.adapter
```

### Context Manager

Temporary configuration overrides.

```python
with dspy.context(lm=other_lm, temperature=0.9):
    # Use different settings temporarily
    result = predict(question="...")
# Original settings restored
```

### Cache Configuration

```python
# Enable/disable cache
dspy.configure_cache(
    enable_disk_cache=True,
    cache_dir=".dspy_cache"
)

# Access cache
dspy.cache.clear()
```

---

## Datasets

**Location:** `dspy/datasets/`

### Dataset

Base class for datasets.

```python
class Dataset:
    @property
    def train(self) -> list[Example]

    @property
    def dev(self) -> list[Example]

    @property
    def test(self) -> list[Example]
```

### Built-in Datasets

```python
# HotPotQA - Multi-hop question answering
dataset = dspy.HotPotQA()
train = dataset.train
dev = dataset.dev

# MATH - Math problem dataset
dataset = dspy.MATH()

# Colors - Simple classification
dataset = dspy.Colors()
```

### DataLoader

Generic data loader.

```python
loader = dspy.DataLoader()
examples = loader.from_csv("data.csv")
examples = loader.from_json("data.json")
```

---

## Utilities

**Location:** `dspy/utils/`

### Async/Sync Conversion

```python
# Convert sync program to async
async_program = dspy.asyncify(sync_program)

# Convert async program to sync
sync_program = dspy.syncify(async_program, in_place=True)
```

### Streaming

```python
# Enable streaming
stream_program = dspy.streamify(program)

# Use stream
for chunk in stream_program(question="..."):
    print(chunk, end="", flush=True)

# Stream listener
class MyListener(dspy.StreamListener):
    def on_chunk(self, chunk):
        print(chunk)

listener = MyListener()
```

### Saving/Loading

```python
# Save state only (predictors' demos/instructions)
module.save("my_program.json", save_program=False)

# Save entire program
module.save("my_program.pkl", save_program=True)

# Load program
loaded = dspy.load("my_program.pkl")
```

### Logging

```python
# Enable logging
dspy.enable_logging()

# Disable logging
dspy.disable_logging()

# Configure loggers
dspy.configure_dspy_loggers(
    level="INFO",
    format="%(asctime)s - %(message)s"
)
```

### Usage Tracking

```python
# Track token usage
with dspy.track_usage() as usage:
    result = predict(question="...")

print(f"Tokens used: {usage.total_tokens}")
print(f"Cost: ${usage.total_cost}")
```

### Inspection

```python
# Inspect LM history
module.inspect_history(n=1)  # Last 1 interaction

# Access history directly
history = module.history
```

---

## Public API Summary

### Main Imports

```python
import dspy

# Core primitives
from dspy import (
    Module, Example, Prediction, Completions,
    BaseModule, PythonInterpreter
)

# Signatures
from dspy import (
    Signature, InputField, OutputField,
    SignatureMeta, make_signature, ensure_signature
)

# Prediction modules
from dspy import (
    Predict, ChainOfThought, ReAct, ProgramOfThought,
    Refine, BestOfN, MultiChainComparison, Parallel,
    CodeAct, KNN, majority
)

# Language models
from dspy import (
    LM, BaseLM, Provider, TrainingJob,
    Embedder, inspect_history
)

# Adapters
from dspy import (
    Adapter, ChatAdapter, JSONAdapter, XMLAdapter,
    TwoStepAdapter, Image, Audio, History, Tool,
    ToolCalls, Code, Type
)

# Optimizers
from dspy import (
    BootstrapFewShot, BootstrapRS, MIPROv2, COPRO,
    LabeledFewShot, BootstrapFinetune, Ensemble,
    KNNFewShot, AvatarOptimizer, GEPA, SIMBA,
    InferRules, BetterTogether, bootstrap_trace_data
)

# Evaluation
from dspy import (
    Evaluate, EM, SemanticF1, CompleteAndGrounded,
    answer_exact_match, normalize_text
)

# Retrievers
from dspy import Retrieve, Embeddings

# Utilities
from dspy import (
    configure, context, settings, cache,
    asyncify, syncify, streamify, load,
    enable_logging, disable_logging, track_usage
)

# Datasets
from dspy import (
    Dataset, DataLoader, HotPotQA, MATH, Colors
)

# Experimental
from dspy import Citations, Document

# Other
from dspy import ColBERTv2
```

---

## Key Patterns & Workflows

### Basic Usage Pattern

```python
# 1. Configure LM
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# 2. Define signature
class QA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# 3. Create predictor
predict = dspy.Predict(QA)
# or
cot = dspy.ChainOfThought(QA)

# 4. Execute
result = predict(question="What is 2+2?")
print(result.answer)
```

### Module Pattern

```python
class RAG(dspy.Module):
    def __init__(self, k=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=k)
        self.qa = dspy.ChainOfThought("question, context -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        context_text = "\n".join([p["long_text"] for p in context])
        return self.qa(question=question, context=context_text)

# Use the module
rag = RAG(k=5)
result = rag(question="What is DSPy?")
```

### Optimization Workflow

```python
# 1. Define metric
def validate_answer(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

# 2. Create program
class MyProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.predictor(question=question)

# 3. Optimize
optimizer = dspy.BootstrapFewShot(
    metric=validate_answer,
    max_bootstrapped_demos=4
)

optimized_program = optimizer.compile(
    student=MyProgram(),
    trainset=train_examples
)

# 4. Evaluate
evaluator = dspy.Evaluate(
    devset=dev_examples,
    metric=validate_answer,
    display_progress=True
)

score = evaluator(optimized_program)
print(f"Score: {score}")
```

### Advanced Optimization with MIPROv2

```python
# MIPROv2 optimizes both instructions and examples
optimizer = dspy.MIPROv2(
    metric=my_metric,
    auto="medium",  # Automatic configuration
    num_candidates=10
)

optimized = optimizer.compile(
    student=MyProgram(),
    trainset=train_examples,
    valset=val_examples,
    num_trials=100,
    max_bootstrapped_demos=4,
    max_labeled_demos=16
)

# Save optimized program
optimized.save("optimized_program.pkl", save_program=True)
```

### Batch Processing

```python
# Process multiple examples in parallel
results = module.batch(
    examples=[ex1, ex2, ex3],
    num_threads=4
)
```

### Streaming Responses

```python
# Enable streaming
stream_predictor = dspy.streamify(predictor)

# Stream output
for chunk in stream_predictor(question="Explain quantum computing"):
    print(chunk, end="", flush=True)
```

### Tool Use with ReAct

```python
# Define tools
def search(query: str) -> str:
    """Search the web for information."""
    return web_search(query)

def calculator(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

# Create ReAct agent
agent = dspy.ReAct(
    signature="question -> answer",
    tools=[search, calculator],
    max_iters=10
)

result = agent(question="What is the GDP of France in 2023 divided by 67 million?")
```

### Context Management

```python
# Default LM
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Use different LM temporarily
with dspy.context(lm=dspy.LM("anthropic/claude-3-5-sonnet-20241022")):
    result = expensive_predictor(question="Complex question")

# Original LM restored here
result = predictor(question="Simple question")
```

---

## Common Use Cases

### Question Answering

```python
class QA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.predictor(question=question)
```

### Retrieval-Augmented Generation

```python
class RAG(dspy.Module):
    def __init__(self, k=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=k)
        self.qa = dspy.ChainOfThought("question, context -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        context_text = "\n".join([p["long_text"] for p in context])
        return self.qa(question=question, context=context_text)
```

### Multi-Hop Reasoning

```python
class MultiHop(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.hop1 = dspy.ChainOfThought("question, context -> search_query")
        self.hop2 = dspy.ChainOfThought("question, context -> answer")

    def forward(self, question):
        # First hop
        context1 = self.retrieve(question).passages
        hop1_result = self.hop1(question=question, context=context1)

        # Second hop
        context2 = self.retrieve(hop1_result.search_query).passages
        return self.hop2(question=question, context=context2)
```

### Classification

```python
class Classify(dspy.Module):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        sig = dspy.Signature(
            "text -> label",
            instructions=f"Classify into: {', '.join(classes)}"
        )
        self.predictor = dspy.Predict(sig)

    def forward(self, text):
        return self.predictor(text=text)
```

### Agent with Tools

```python
class Agent(dspy.Module):
    def __init__(self, tools):
        super().__init__()
        self.react = dspy.ReAct(
            "task -> result",
            tools=tools,
            max_iters=10
        )

    def forward(self, task):
        return self.react(task=task)
```

---

## Best Practices

1. **Start Simple**: Begin with `Predict` or `ChainOfThought`, optimize later
2. **Use Signatures**: Define clear input/output schemas with descriptions
3. **Optimize Systematically**: Use optimizers like `BootstrapFewShot` or `MIPROv2`
4. **Evaluate Thoroughly**: Create good metrics and validation sets
5. **Cache Wisely**: Enable caching for expensive operations
6. **Track Usage**: Monitor token usage and costs in production
7. **Save Programs**: Save optimized programs to avoid re-optimization
8. **Use Context Managers**: Temporarily override settings when needed
9. **Handle Errors**: Use `max_errors` in evaluators and optimizers
10. **Profile First**: Use inspection tools to understand LM behavior

---

## Additional Resources

- **Repository:** https://github.com/stanfordnlp/dspy
- **Documentation:** https://dspy-docs.vercel.app/
- **Examples:** https://github.com/stanfordnlp/dspy/tree/main/examples
- **Discord:** https://discord.gg/VzS6RHHK6F

---

*This reference guide was generated for DSPy v3.0.4b2 on December 16, 2025.*
