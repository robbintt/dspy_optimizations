# MDAP (Massively Decomposed Agentic Processes) Harness

Implementation of the MAKER framework: Maximal Agentic decomposition, first-to-ahead-by-K Error correction, and Red-flagging for reliable long-horizon LLM task execution.

## Overview

MDAP enables Large Language Models to reliably execute tasks spanning over one million dependent steps with zero errors by combining extreme task decomposition with statistical error correction. This framework transforms tasks that are statistically impossible for a single model into solvable engineering problems.

### Core Components

1. **Maximal Agentic Decomposition (MAD)**: Break tasks into smallest atomic subtasks (m=1)
2. **First-to-ahead-by-K Voting**: Dynamic sampling until one answer leads by K votes
3. **Red-flagging**: Pre-voting filter that discards invalid responses

## Architecture

```
MDAP Harness
├── mdap_harness.py      # Core framework implementation
├── hanoi_solver.py      # Towers of Hanoi example implementation
├── test_hanoi.py        # Test suite and benchmarks
├── example_hanoi.py     # Simple usage example
├── setup_mdap.sh        # Environment setup script
├── run_mdap.sh          # Execution script with convenience commands
├── requirements_mdap.txt # Python dependencies
└── .env.example         # Environment variables template
```

## Quick Start

### 1. Setup Environment

```bash
# Make setup script executable
chmod +x setup_mdap.sh

# Run setup (creates virtual environment, installs dependencies)
./setup_mdap.sh

# Activate virtual environment
source venv_mdap/bin/activate
```

### 2. Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
# Required: OPENAI_API_KEY or other provider key
```

### 3. Run Examples

```bash
# Make run script executable
chmod +x run_mdap.sh

# Run simple 3-disk example
./run_mdap.sh example 3

# Run full test suite
./run_mdap.sh test

# Solve specific problem
./run_mdap.sh solve 4
```

## Core Classes

### MDAPConfig

Configuration class for MDAP execution parameters:

```python
@dataclass
class MDAPConfig:
    model: str = "gpt-4o-mini"           # LLM model to use
    k_margin: int = 3                    # First-to-ahead-by-K margin
    max_candidates: int = 10             # Max candidates to sample
    temperature: float = 0.1             # Sampling temperature
    max_retries: int = 3                 # Retry attempts per step
    cost_threshold: Optional[float] = None  # Cost limit
```

### MDAPHarness

Main framework class implementing MAKER components:

```python
class MDAPHarness:
    def __init__(self, config: MDAPConfig)
    async def first_to_ahead_by_k(self, prompt: str, response_parser: Callable) -> Any
    async def execute_step(self, step_prompt: str, response_parser: Callable) -> Any
    async def execute_mdap(self, initial_state: Any, step_generator: Callable, termination_check: Callable) -> List[Any]
    def update_state(self, current_state: Any, step_result: Any) -> Any
```

### RedFlagParser

Filters invalid responses before voting:

```python
class RedFlagParser:
    @staticmethod
    def parse_move_state_flag(response: str) -> Optional[Dict[str, Any]]
```

## Implementation Details

### First-to-ahead-by-K Voting

The voting mechanism samples candidates until one response leads by K votes:

1. Generate candidate responses using LLM
2. Apply red-flagging to filter invalid responses
3. Count votes for valid responses
4. Continue until one response leads by K votes or max candidates reached
5. Return winner or majority vote

### Red-flagging Rules

Responses are filtered based on:

1. **Length**: Reject overly long responses (>500 chars)
2. **Format**: Must be valid JSON/dictionary
3. **Required Fields**: All critical fields must exist and not be None
4. **Valid Values**: Field values must be in allowed sets
5. **Logic**: Cannot move from source to same destination

### State Management

The framework maintains:
- `current_state`: Current problem state
- `execution_trace`: Complete history of states
- `step_count`: Number of steps executed
- `total_cost`: Accumulated API costs

## Towers of Hanoi Example

### HanoiState

```python
@dataclass
class HanoiState:
    pegs: dict          # {'A': [largest...smallest], 'B': [...], 'C': [...]}
    num_disks: int      # Total number of disks
    move_count: int = 0 # Number of moves made
```

### HanoiMDAP

Specialized implementation for Towers of Hanoi:

```python
class HanoiMDAP(MDAPHarness):
    def create_initial_state(self, num_disks: int) -> HanoiState
    def generate_step_prompt(self, state: HanoiState) -> str
    def is_valid_move(self, state: HanoiState, from_peg: str, to_peg: str) -> bool
    def update_state(self, current_state: HanoiState, step_result: dict) -> HanoiState
    def is_solved(self, state: HanoiState) -> bool
    async def solve_hanoi(self, num_disks: int) -> List[HanoiState]
```

## Usage Examples

### Basic Usage

```python
import asyncio
from hanoi_solver import HanoiMDAP, MDAPConfig

async def main():
    # Create solver with custom config
    config = MDAPConfig(
        model="gpt-4o-mini",
        k_margin=3,
        max_candidates=10,
        temperature=0.1
    )
    
    solver = HanoiMDAP(config)
    
    # Solve 3-disk Hanoi
    trace = await solver.solve_hanoi(3)
    
    # Check results
    final_state = trace[-1]
    print(f"Solved in {final_state.move_count} moves")

asyncio.run(main())
```

### Custom Implementation

```python
class CustomMDAP(MDAPHarness):
    def update_state(self, current_state, step_result):
        # Implement your state update logic
        new_state = current_state.copy()
        # ... update logic ...
        return new_state
    
    def step_generator(self, state):
        prompt = f"Generate prompt based on {state}"
        parser = self.custom_parser
        return prompt, parser
    
    def custom_parser(self, response):
        # Implement your response parsing logic
        # Return None to red-flag invalid responses
        return parsed_data
```

## Configuration

### Environment Variables

Create `.env` file from `.env.example`:

```bash
# Required: API key for your LLM provider
OPENAI_API_KEY="your-api-key-here"

# Optional: Alternative providers
# ANTHROPIC_API_KEY="your-anthropic-api-key-here"
# AZURE_API_KEY="your-azure-api-key-here"

# LiteLLM Configuration
LITELLM_LOG="INFO"  # Set to DEBUG for verbose logging

# MDAP Configuration (optional overrides)
# MDAP_DEFAULT_MODEL="gpt-4o-mini"
# MDAP_K_MARGIN="3"
# MDAP_MAX_CANDIDATES="10"
# MDAP_TEMPERATURE="0.1"
```

### Supported Providers

MDAP uses LiteLLM for unified API access. Supported providers include:
- OpenAI (GPT-3.5, GPT-4, GPT-4o, etc.)
- Anthropic (Claude)
- Azure OpenAI
- Google (Gemini)
- And many more via LiteLLM

## Testing

### Run Test Suite

```bash
# Run all tests
./run_mdap.sh test

# Run specific disk count test
./run_mdap.sh test 3
```

### Benchmark Performance

```bash
# Benchmark different K margins
./run_mdap.sh benchmark

# Benchmark with specific disk count
./run_mdap.sh benchmark 4
```

### Test Cases

The test suite includes:
- 3-disk Hanoi (7 moves optimal)
- 4-disk Hanoi (15 moves optimal)
- Performance benchmarks with different K margins
- Error handling validation

## Performance Considerations

### Cost vs Reliability

- **K Margin**: Higher K increases reliability but also cost
- **Model Choice**: Smaller models (gpt-4o-mini) are cost-effective with error correction
- **Temperature**: Lower temperature (0.1) recommended for consistent responses

### Scaling

For long-horizon tasks:
- Use `k_margin = Θ(ln(steps))` for optimal scaling
- Monitor cost with `cost_threshold` parameter
- Consider batch processing for parallel steps

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure `.env` file contains valid API key
2. **Virtual Environment**: Activate with `source venv_mdap/bin/activate`
3. **Dependencies**: Re-run `./setup_mdap.sh` if imports fail
4. **Memory Issues**: Reduce `max_candidates` for large problems

### Debug Mode

Enable verbose logging:

```bash
# Set in .env file
LITELLM_LOG="DEBUG"

# Or export directly
export LITELLM_LOG="DEBUG"
./run_mdap.sh example 3
```

## Contributing

### Adding New Problems

1. Create state representation class
2. Implement MDAPHarness subclass
3. Define step_generator and update_state methods
4. Add termination_check function
5. Create test cases

### Example Structure

```python
@dataclass
class ProblemState:
    # Define your state structure
    pass

class ProblemMDAP(MDAPHarness):
    def update_state(self, current_state, step_result):
        # Implement state transition logic
        pass
    
    def step_generator(self, state):
        # Generate prompt and parser for current step
        pass
    
    def is_solved(self, state):
        # Check termination condition
        pass
```

## License

This implementation follows the MAKER framework research principles for reliable long-horizon LLM task execution.

## References

- MAKER Framework: Maximal Agentic decomposition, first-to-ahead-by-K Error correction, and Red-flagging
- LiteLLM: Unified LLM API access
- Towers of Hanoi: Classic recursive problem for demonstration
