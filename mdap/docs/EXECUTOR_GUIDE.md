# MicroAgent Executor Guide

## Overview

The `MicroAgentExecutor` provides a generic, reusable way to execute any `MicroAgent` implementation with the MDAP framework. It separates execution concerns from agent logic, making it easier to create and run new agent types.

## Why Use the Executor?

### Before (Direct Harness Usage)

```python
class MyAgent(MicroAgent):
    def __init__(self, config):
        super().__init__(config)
        self.harness = MDAPHarness(config)  # Each agent manages its own harness

    async def solve(self, *args):
        trace = await self.harness.execute_agent_mdap(self, *args)
        return trace
```

Problems:
- Each agent needs to create and manage its own harness
- Boilerplate code duplicated across agents
- Statistics tracking requires custom implementation
- Calibration logic must be reimplemented

### After (Executor Pattern)

```python
class MyAgent(MicroAgent):
    # Just implement the required methods - no harness management needed
    def create_initial_state(self, *args): ...
    def generate_step_prompt(self, state): ...
    def update_state(self, current_state, step_result): ...
    def is_solved(self, state): ...
```

```python
# Usage
agent = MyAgent()
executor = MicroAgentExecutor(agent)
trace = await executor.execute(*args)
```

Benefits:
- Clean separation of concerns
- Built-in statistics tracking
- Unified calibration interface
- Reusable across all agent types
- Less boilerplate code

## Basic Usage

### 1. Simple Execution

```python
from mdap.microagent_executor import MicroAgentExecutor
from mdap.hanoi_solver import HanoiMDAP

# Create agent
agent = HanoiMDAP()

# Create executor
executor = MicroAgentExecutor(agent)

# Execute
trace = await executor.execute(num_disks=5)

# Access results
print(f"Solved in {trace[-1].move_count} moves")
print(f"Total cost: ${executor.total_cost:.4f}")
```

### 2. One-Line Convenience Function

```python
from mdap.microagent_executor import execute_agent
from mdap.hanoi_solver import HanoiMDAP

trace = await execute_agent(HanoiMDAP(), num_disks=5)
```

### 3. Custom Configuration

```python
from mdap.mdap_harness import MDAPConfig

config = MDAPConfig(
    k_margin=5,
    max_candidates=15,
    temperature=0.7
)

agent = HanoiMDAP(config)
executor = MicroAgentExecutor(agent, config)
trace = await executor.execute(num_disks=5)
```

## Advanced Features

### Statistics Tracking

```python
executor = MicroAgentExecutor(agent)
trace = await executor.execute(num_disks=5)

# Individual properties
print(f"Cost: ${executor.total_cost:.4f}")
print(f"API calls: {executor.total_api_calls}")
print(f"Input tokens: {executor.total_input_tokens}")
print(f"Output tokens: {executor.total_output_tokens}")

# Complete statistics
stats = executor.get_statistics()
print(stats)
# {
#     'total_cost': 0.0234,
#     'total_api_calls': 42,
#     'total_input_tokens': 1520,
#     'total_output_tokens': 380,
#     'model': 'cerebras/zai-glm-4.6',
#     'k_margin': 3,
#     'max_candidates': 10,
#     'temperature': 0.6
# }
```

### Multiple Runs with Statistics Reset

```python
executor = MicroAgentExecutor(agent)

for num_disks in [3, 4, 5, 6]:
    executor.reset_statistics()
    trace = await executor.execute(num_disks=num_disks)

    print(f"{num_disks} disks: {trace[-1].move_count} moves, "
          f"${executor.total_cost:.4f}, "
          f"{executor.total_api_calls} calls")
```

### Calibration

```python
# Load or generate calibration states
calibration_states = load_calibration_cache()

# Run calibration
executor = MicroAgentExecutor(agent)
results = await executor.calibrate(calibration_states)

print(f"Per-step success rate: {results['p_estimate']:.4f}")
print(f"Recommended k_min: {results['k_min']}")
print(f"Current k_margin: {results['current_k_margin']}")

# Update k_margin if needed
if results['k_min'] > executor.config.k_margin:
    executor.update_k_margin(results['k_min'])
    print(f"Updated k_margin to {results['k_min']}")
```

### Dynamic Parameter Tuning

```python
executor = MicroAgentExecutor(agent)

# Start with conservative k_margin
executor.update_k_margin(5)
trace1 = await executor.execute(num_disks=5)

# Reduce for faster execution if confident
executor.update_k_margin(3)
trace2 = await executor.execute(num_disks=4)
```

## Creating New Agent Types

To create a new agent type that works with the executor:

```python
from mdap.microagent import MicroAgent
from dataclasses import dataclass

@dataclass
class MyProblemState:
    # Your state representation
    data: dict
    step_count: int = 0

class MyProblemAgent(MicroAgent):
    def create_initial_state(self, problem_input):
        return MyProblemState(data=problem_input)

    def generate_step_prompt(self, state):
        # Generate LLM prompt based on state
        return f"Given {state.data}, what's the next step?"

    def update_state(self, current_state, step_result):
        # Update state with LLM's response
        new_state = current_state.copy()
        new_state.data.update(step_result)
        new_state.step_count += 1
        return new_state

    def is_solved(self, state):
        # Check if problem is solved
        return state.data.get('solved', False)

    def step_generator(self, state):
        # Return (prompt, parser) tuple
        prompt = self.generate_step_prompt(state)
        parser = self.get_response_parser()
        return prompt, parser

    def get_response_parser(self):
        # Return a function that parses LLM responses
        def parse(response):
            # Your parsing logic
            return json.loads(response)
        return parse
```

Then use it with the executor:

```python
agent = MyProblemAgent()
executor = MicroAgentExecutor(agent)
trace = await executor.execute(problem_input={'x': 1, 'y': 2})
```

## Architecture

```
┌─────────────────────────────────────────┐
│         MicroAgentExecutor              │
│  ┌───────────────────────────────────┐  │
│  │  Configuration & Initialization   │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │      MDAPHarness Instance         │  │
│  │  (Handles voting & red-flagging)  │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │    Statistics & Monitoring        │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
                  │
                  │ executes
                  ▼
┌─────────────────────────────────────────┐
│          MicroAgent (Abstract)          │
│  ┌───────────────────────────────────┐  │
│  │  create_initial_state()           │  │
│  │  generate_step_prompt()           │  │
│  │  update_state()                   │  │
│  │  is_solved()                      │  │
│  │  step_generator()                 │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
                  △
                  │ implements
                  │
     ┌────────────┴────────────┐
     │                         │
┌─────────┐            ┌────────────┐
│ Hanoi   │            │  Custom    │
│ Agent   │            │  Agents    │
└─────────┘            └────────────┘
```

## Comparison: Executor vs Direct Harness

| Aspect | MicroAgentExecutor | Direct Harness |
|--------|-------------------|----------------|
| **Setup** | `executor = MicroAgentExecutor(agent)` | `harness = MDAPHarness(config)` |
| **Execution** | `trace = await executor.execute(*args)` | `trace = await harness.execute_agent_mdap(agent, *args)` |
| **Statistics** | Built-in properties & methods | Manual tracking required |
| **Calibration** | `executor.calibrate(states)` | `harness.estimate_per_step_success_rate_from_states(...)` |
| **Config Updates** | `executor.update_k_margin(5)` | `harness.config.k_margin = 5` |
| **Use Case** | High-level agent execution | Low-level control, custom workflows |

## Best Practices

1. **Use the executor for standard workflows**: If you're just running an agent end-to-end, use the executor.

2. **Reset statistics between runs**: When running multiple problems, reset stats for accurate per-run tracking.

3. **Calibrate before production**: Run calibration to determine optimal `k_margin` for your model and problem.

4. **Monitor costs**: Check `executor.total_cost` regularly, especially during development.

5. **Use the convenience function for scripts**: For simple scripts, `execute_agent()` is the quickest way to run an agent.

6. **Keep agents focused**: Let agents handle problem logic, let the executor handle execution mechanics.

## Examples

See `example_executor.py` for complete working examples of all patterns described above.

## Testing

Run the executor tests:

```bash
pytest test_microagent_executor.py -v
```
