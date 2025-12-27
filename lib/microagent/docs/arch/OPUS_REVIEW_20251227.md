# MicroAgent Framework Architecture Review

**Date:** 2025-12-27
**Reviewer:** Claude Opus 4.5
**Goal:** Assess separation of concerns to inform DeclarativeAgent design

---

## 1. Overall Structure

The `lib/microagent` library consists of 4 core Python files:

```
lib/microagent/
├── microagent/
│   ├── __init__.py
│   ├── microagent.py          # Abstract base class
│   ├── protocols.py           # ExecutionHarness protocol
│   ├── microagent_executor.py # Orchestrator
│   └── litellm_harness.py     # LLM integration
├── pyproject.toml
└── .venv/
```

**Package Metadata:**
- Name: `microagent` (version 0.1.0)
- Dependencies: `litellm`, `pyyaml`
- Requires: Python >= 3.10

---

## 2. Component Responsibilities

### 2.1 MicroAgent (Abstract Base Class)

**File:** `microagent/microagent.py`

**Responsibility:** Defines the problem domain contract for solving step-by-step problems using LLMs.

**Core Abstract Methods:**

| Method | Lines | Purpose |
|--------|-------|---------|
| `create_initial_state(*args, **kwargs)` | 19-21 | Creates problem's starting state |
| `generate_step_prompt(state)` | 24-26 | Generates LLM prompt for current step |
| `update_state(current_state, step_result)` | 29-31 | Applies step result to state |
| `is_solved(state)` | 34-36 | Termination condition check |
| `get_problem_complexity(state)` | 39-49 | Returns problem complexity metric |

**Template Methods:**

| Method | Lines | Purpose |
|--------|-------|---------|
| `step_generator(state)` | 51-59 | Returns `(prompt, parser)` tuple |
| `get_response_parser()` | 61-66 | Default passthrough parser |
| `validate_step_result(step_result)` | 68-73 | Optional validation hook |

**Configuration:**
- Accepts optional `config` parameter in `__init__` (line 14-16)
- Stored as `self.config` for subclass use
- No type enforcement

---

### 2.2 ExecutionHarness (Protocol)

**File:** `microagent/protocols.py`

**Responsibility:** Defines execution contract - decouples agent logic from LLM interaction strategy.

**Abstract Methods:**

```python
def execute_step(
    step_prompt: Tuple[str, str],  # (system_prompt, user_prompt)
    response_parser: Callable
) -> Any
```
- Single atomic step execution (lines 18-31)

```python
def execute_plan(
    initial_state: Any,
    step_generator: Callable,
    termination_check: Callable,
    agent: Any
) -> List[Any]
```
- Full problem solution loop (lines 34-51)

**Known Implementations:**
1. `LiteLLMHarness` - Basic litellm integration
2. `MDAPHarness` - MDAP framework with voting/red-flagging
3. `HelloWorldHarness` - Simple adapter example

---

### 2.3 MicroAgentExecutor

**File:** `microagent/microagent_executor.py`

**Responsibility:** Dependency injection orchestrator connecting agent + harness.

**Implementation (lines 50-81):**

```python
async def execute(self, *args, **kwargs):
    initial_state = self.agent.create_initial_state(*args, **kwargs)
    trace = await self.harness.execute_plan(
        initial_state,
        self.agent.step_generator,
        self.agent.is_solved,
        self.agent
    )
    return trace
```

**Critical Observation:** The executor is essentially a pass-through. It:
1. Calls `agent.create_initial_state()`
2. Delegates everything else to `harness.execute_plan()`

---

### 2.4 LiteLLMHarness

**File:** `microagent/litellm_harness.py`

**LiteLLMConfig (lines 19-76):**
- Configuration management with YAML/dict support
- Hierarchical precedence: kwargs > model_config > defaults > fallback

**LiteLLMHarness (lines 79-189):**

`execute_step()` (lines 108-167):
- Takes `(system_prompt, user_prompt)` tuple and parser
- Implements exponential backoff retries with jitter
- Tracks cost from `response._hidden_params['response_cost']`

`execute_plan()` (lines 169-188):
```python
while not termination_check(current_state):
    prompt, parser = step_generator(current_state)
    step_result = await self.execute_step(prompt, parser)
    current_state = agent.update_state(current_state, step_result)
    trace.append(current_state)
return trace
```

**Key Issue:** The harness owns the entire execution loop, not the executor.

---

## 3. Component Interaction Flow

```
User Code
    │
    ▼
MicroAgentExecutor.execute()
    ├─► agent.create_initial_state() → initial_state
    └─► harness.execute_plan(initial_state, callbacks...)
            │
            ▼
        Loop: while not agent.is_solved(state):
            ├─► agent.step_generator(state)
            │   ├─► agent.generate_step_prompt(state)
            │   └─► agent.get_response_parser()
            │   └─► returns: ((system, user), parser)
            │
            ├─► harness.execute_step(prompt_tuple, parser)
            │   ├─► LLM call with retries
            │   └─► return parser(raw_result)
            │
            └─► agent.update_state(state, step_result)
                └─► returns: new_state
```

**Callback Pattern:** Agent provides four callbacks to harness:
1. `step_generator` - produces prompts
2. `is_solved` - termination predicate
3. `update_state` - state transition
4. `get_problem_complexity` - calibration metric

---

## 4. Separation of Concerns Analysis

### 4.1 Well-Separated Concerns

| Aspect | Status | Notes |
|--------|--------|-------|
| Agent-Executor Separation | ✅ Good | Agent defines domain logic only |
| Configuration Abstraction | ✅ Good | LiteLLMConfig isolates YAML parsing |
| Protocol-Based Polymorphism | ✅ Good | Multiple harness implementations |

### 4.2 Mixed/Unclear Concerns

#### Issue 1: Executor Is Nearly Unnecessary

**Location:** `microagent_executor.py:50-81`

The executor's only function is:
1. Call `agent.create_initial_state()`
2. Delegate to `harness.execute_plan()`

This raises the question: why have an executor at all?

#### Issue 2: Harness Owns the Execution Loop

**Location:** `litellm_harness.py:169-188`

The harness implements the full `while not solved` loop. This means:
- Harness knows about agent callbacks (`step_generator`, `is_solved`, `update_state`)
- Harness orchestrates state progression
- Different harnesses can have different loop behaviors (LiteLLM vs MDAP)

**Consequence:** The "harness" is really an "executor with LLM integration."

#### Issue 3: Agent Creates Own Dependencies

**Location:** `mdap/hanoi_solver.py:121`

```python
class HanoiMDAP(MicroAgent):
    def __init__(self, config: MDAPConfig = None):
        super().__init__(config)
        self.harness = MDAPHarness(self.config)  # ← Violates DI!
```

Agent instantiates its own harness, bypassing executor injection.

#### Issue 4: Unused Validation Hook

**Location:** `microagent.py:68-73`

```python
def validate_step_result(self, step_result: Any) -> bool:
    """Validate step result before updating state"""
    return step_result is not None
```

This method exists but is never called by any harness implementation.

#### Issue 5: Callback Proliferation

Agent provides 4 separate callbacks. The harness must know to call each one in the right order. This creates tight coupling despite the protocol abstraction.

#### Issue 6: Mixed Prompt/Parser Responsibility

`step_generator()` returns both `(prompt_tuple, parser)`. These are conceptually separate:
- Prompt generation = input concern
- Response parsing = output concern

Bundling them means you can't configure parsing independently.

#### Issue 7: Configuration Type Unsafety

**LiteLLMHarness** (line 92-93):
```python
if not isinstance(config, LiteLLMConfig):
    raise TypeError("config must be an instance of LiteLLMConfig")
```

**MicroAgent** (line 14-16):
```python
def __init__(self, config=None):
    self.config = config  # No type checking!
```

Inconsistent defensive programming.

---

## 5. Responsibility Matrix

| Responsibility | MicroAgent | Executor | Harness |
|----------------|------------|----------|---------|
| Problem state definition | ✅ | | |
| Prompt generation | ✅ | | |
| Response parsing | ✅ | | |
| State updates | ✅ | | |
| Termination check | ✅ | | |
| Initial state creation | ✅ | (calls) | |
| Execution loop | | | ✅ |
| LLM interaction | | | ✅ |
| Retry logic | | | ✅ |
| Cost tracking | | | ✅ |
| Dependency injection | | ✅ | |

**Observation:** Executor's only unique responsibility is dependency injection. Everything else is either agent or harness.

---

## 6. Code Duplication

### 6.1 Configuration Hierarchy Pattern

**Location:** `litellm_harness.py:64-68`

```python
self.temperature = kwargs.get('temperature',
                             model_config.get('temperature',
                             defaults.get('temperature', 0.7)))
```

This pattern repeats 5 times. Could be extracted:
```python
def _get_config_value(self, key, kwargs, model_config, defaults, fallback):
    return kwargs.get(key, model_config.get(key, defaults.get(key, fallback)))
```

### 6.2 Retry Logic

Both `LiteLLMHarness.execute_step()` and `MDAPHarness.execute_step()` implement similar exponential backoff. Could be shared utility.

---

## 7. Architecture Options for DeclarativeAgent

### Option A: Keep Executor, Slim Down Harness

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│ DeclarativeAgent│ ──► │ MicroAgentExecutor│ ──► │   Harness   │
│   (YAML-driven) │     │  (owns loop)     │     │ (LLM only)  │
└─────────────────┘     └──────────────────┘     └─────────────┘
```

- Move execution loop from harness → executor
- Harness becomes pure `execute_step()` (single LLM call)
- Executor orchestrates state progression
- DeclarativeAgent replaces MicroAgent with YAML config

**Pros:**
- Clean separation of orchestration vs LLM interaction
- Harness becomes truly swappable
- Executor has a real purpose

**Cons:**
- Breaking change to harness protocol
- MDAP-specific loop logic would need rework

### Option B: Merge Executor Into Harness

```
┌─────────────────┐     ┌─────────────────────────┐
│ DeclarativeAgent│ ──► │ ExecutorHarness         │
│   (YAML-driven) │     │ (orchestration + LLM)   │
└─────────────────┘     └─────────────────────────┘
```

- Eliminate MicroAgentExecutor
- Harness IS the executor
- DeclarativeAgent configures via YAML

**Pros:**
- Simpler architecture (one less layer)
- Matches current de facto behavior

**Cons:**
- Harness becomes harder to swap
- Mixing orchestration with LLM concerns

### Option C: Invert Control (Agent-Centric)

```
┌─────────────────────────────────────┐
│ DeclarativeAgent                    │
│   ├── owns execution loop           │
│   ├── calls harness.execute_step()  │
│   └── configured by YAML            │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────┐
│     Harness     │
│   (LLM only)    │
└─────────────────┘
```

- Agent owns its own execution
- No executor needed
- Harness is pure LLM abstraction

**Pros:**
- Agent has full control
- Simpler dependency graph
- Natural fit for declarative config

**Cons:**
- Agents become heavier
- Harder to share orchestration logic

---

## 8. Recommendations

### Immediate Fixes

1. **Remove unused `validate_step_result()` hook** or make harnesses call it
2. **Fix HanoiMDAP dependency injection** - don't create harness inside agent
3. **Add type annotations** to `MicroAgent.config` parameter

### For DeclarativeAgent Design

1. **Choose Option A or C** - both give harness a clear, limited role
2. **Separate prompt templates from parsing rules** in YAML config
3. **Define state schema declaratively** with JSON Schema or similar
4. **Use Jinja2 or similar** for prompt templating with state interpolation
5. **Define termination conditions** as expressions (e.g., `state.remaining == 0`)

### Suggested YAML Structure

```yaml
agent:
  name: "example-declarative-agent"

state:
  schema:
    type: object
    properties:
      remaining: { type: integer }
      result: { type: string }
  initial:
    remaining: 10
    result: ""

prompts:
  system: "You are a helpful assistant."
  user: "Process item {{ state.remaining }}. Previous: {{ state.result }}"

parsing:
  pattern: "Result: (.+)"
  extract: "result"

transitions:
  update:
    remaining: "{{ state.remaining - 1 }}"
    result: "{{ parsed.result }}"

termination:
  condition: "state.remaining == 0"
```

---

## 9. Summary

| Component | Current State | Recommendation |
|-----------|--------------|----------------|
| MicroAgent | Clean abstraction | Keep, add type hints |
| Executor | Pass-through only | Either expand role or eliminate |
| Harness | Does too much | Reduce to LLM-only |
| Config | Works but duplicated | Extract hierarchy helper |

The framework is reasonably well-designed but has role ambiguity between executor and harness. For DeclarativeAgent, the key insight is that most abstract methods could be YAML-driven: prompt templates, parsing patterns, state schema, and termination conditions.
