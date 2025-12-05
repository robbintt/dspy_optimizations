# DSPy 3.0 Technical Specification

## 1\. Comparative Mode (The Migration Path)

### Model Instantiation: Unified `dspy.LM` Client

DSPy 3.0 deprecates provider-specific clients (e.g., `dspy.OpenAI`, `dspy.HFClient`, `dspy.Anthropic`) in favor of a single, unified `dspy.LM` class that interfaces with LiteLLM.

| Feature | DSPy v2.x (Legacy) | DSPy v3.x (Current) |
| :--- | :--- | :--- |
| **Client** | `dspy.OpenAI`, `dspy.Anthropic` | `dspy.LM` |
| **Instantiation** | `lm = dspy.OpenAI(model='gpt-4')` | `lm = dspy.LM('openai/gpt-4o')` |
| **Provider Spec** | Implicit or Class-based | Explicit prefix string (e.g., `openai/`, `anthropic/`) |
| **Arguments** | `api_key`, `model_type` vary by class | Unified `model`, `api_key`, `api_base`, `temperature` |

**Migration Example:**

```python
# v2.x
lm = dspy.OpenAI(model='gpt-4', api_key="...")

# v3.x
lm = dspy.LM(
    model='openai/gpt-4o',
    api_key="...",
    temperature=0.0
)
```

### Configuration: `dspy.settings.configure`

Global configuration now accepts the unified `lm` object and a new `adapter` argument.

```python
# v2.x
dspy.settings.configure(lm=lm)

# v3.x
dspy.settings.configure(
    lm=lm,
    adapter=dspy.ChatAdapter(), # New in v3: Explicit Adapter control
    experimental=True # often needed for new optimizers
)
```

### Optimizers: BootstrapFewShot vs. MIPROv2

MIPROv2 is the primary optimizer in v3, replacing the need for complex manual pipelines. It uses Bayesian Optimization to generate data-aware instructions and demonstrations.

| Optimizer | `BootstrapFewShot` (Legacy/Simple) | `MIPROv2` (v3 Standard) |
| :--- | :--- | :--- |
| **Mechanism** | Randomly selects/bootstraps examples. | Optimizes **instructions** and **demonstrations** jointly using Bayesian Optimization. |
| **Cost** | Low (few calls). | Higher (requires `auto` tuning or explicit trial count). |
| **Usage** | Best for low-data, fixed-instruction tasks. | Best for maximizing performance on complex tasks. |

**Migration Example:**

```python
# v2.x BootstrapFewShot
optimizer = dspy.BootstrapFewShot(metric=my_metric, max_bootstrapped_demos=4)
program = optimizer.compile(module, trainset=trainset)

# v3.x MIPROv2
optimizer = dspy.MIPROv2(
    metric=my_metric,
    auto="light", # Options: "light", "medium", "heavy"
    num_threads=24
)
program = optimizer.compile(
    module,
    trainset=trainset,
    max_bootstrapped_demos=2,
    max_labeled_demos=2
)
```

## 2\. Mechanistic Mode (The New Specs)

### `dspy.LM` Class Definition

The unified client for all LLM interactions.

```python
class dspy.LM(dspy.BaseLM):
    def __init__(
        self,
        model: str, # "provider/model_name" format (e.g. "openai/gpt-4o")
        model_type: Literal['chat', 'text', 'responses'] = 'chat',
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,
        provider: Provider | None = None,
        finetuning_model: str | None = None,
        api_base: str | None = None, # Passed via kwargs
        api_key: str | None = None,  # Passed via kwargs
        **kwargs
    ):
        ...

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs
    ) -> list[dict[str, Any] | str]:
        """Synchronous execution."""
        ...

    async def acall(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs
    ) -> list[dict[str, Any] | str]:
        """Asynchronous execution."""
        ...
```

### Adapters

Adapters translate DSPy Signatures into provider-specific message formats (e.g., list of dicts for Chat APIs).

#### `dspy.ChatAdapter` (Default)

Handles standard chat interactions with fallback mechanisms.

```python
class dspy.ChatAdapter(dspy.Adapter):
    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False, # Set True for tool use
        native_response_types: list[type] | None = None,
        use_json_adapter_fallback: bool = True
    ):
        ...
```

#### `dspy.JSONAdapter`

Forces structured JSON output, often used as a fallback or for strict schema requirements.

```python
class dspy.JSONAdapter(dspy.ChatAdapter):
    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = True # Defaults to True
    ):
        ...
```

### Signatures

Signatures remain the declarative contract but now strongly support `pydantic` models and typed fields via `dspy.InputField` and `dspy.OutputField`.

```python
class RAGSignature(dspy.Signature):
    """Retrieves context and answers the question."""
    
    # Input Field with description
    context: list[str] = dspy.InputField(desc="Relevant documents")
    question: str = dspy.InputField()
    
    # Output Field with description
    answer: str = dspy.OutputField(desc="Concise answer based on context")

# Inline usage remains supported
sig = dspy.Signature("context, question -> answer")
```

## 3\. Synthesizing Mode (The Feature Set)

  * **Native Tool Use**:

      * **Class**: `dspy.Tool(func, name=..., desc=..., args=...)`.
      * **Usage**: Pass `tools=[my_tool]` to modules like `dspy.ReAct`.
      * **Async**: Supports `acall` for async tool execution.
      * **Integration**: `ChatAdapter` with `use_native_function_calling=True` maps these to LLM native function calling APIs (e.g., OpenAI Tools).

  * **Observability & Tracking**:

      * **MLflow**: Native integration for logging traces and artifacts.
      * **Callbacks**: `dspy.LM` and Adapters accept `callbacks` lists for custom logging/tracing hooks.
      * **Usage Tracking**: `dspy.settings.usage_tracker` allows monitoring token usage and cost across calls.

  * **Async Support**:

      * **Pattern**: `acall` (async call) methods are now pervasive across `dspy.LM`, `dspy.Module`, `dspy.Adapter`, and `dspy.Tool`.
      * **Execution**: `await module.acall(...)` allows for non-blocking concurrent execution of DSPy programs.
