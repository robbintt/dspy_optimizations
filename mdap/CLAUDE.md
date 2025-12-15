# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MDAP (Massively Decomposed Agentic Processes) is an implementation of the MAKER framework for solving multi-step problems using LLMs with error correction. It implements:

- **M**aximal Agentic decomposition - Breaking problems into atomic steps
- **first-to-A**head-by-**K** **E**rror correction - Voting mechanism where sampling continues until one answer leads by K votes
- **R**ed-flagging - Filtering invalid responses before voting

The primary demonstration is solving Towers of Hanoi problems using LLM-guided step-by-step execution.

## Commands

### Setup
```bash
./setup_mdap.sh                    # Initial environment setup
source ~/virtualenvs/mdap_harness_venv/bin/activate  # Activate venv
```

### Running the Solver
```bash
./run_mdap.sh example <num_disks>           # Run Hanoi solver
./run_mdap.sh example <num_disks> --k <k>   # Run with custom k_margin
MDAP_MOCK_MODE=true ./run_mdap.sh example 3 # Run in mock mode (no API calls)
```

### Calibration
```bash
./calibrate_hanoi.sh                        # Run calibration (regenerates cache)
./calibrate_hanoi.sh --use_cache            # Use existing calibration cache
./calibrate_hanoi.sh --sample_steps 20      # Custom sample size
python extract_calibration_data.py          # Generate analysis report
```

### Testing
```bash
./test_mdap.sh unit          # Run unit tests only
./test_mdap.sh integration   # Run integration tests only
./test_mdap.sh calibration   # Run calibration tests only
./test_mdap.sh all           # Run all tests
./test_mdap.sh coverage      # Run with coverage report
./test_mdap.sh specific test_mdap_harness.py::TestRedFlagParser  # Single test
```

## Architecture

### Core Components

**`mdap_harness.py`** - Main harness implementing MAKER framework:
- `MDAPConfig`: Configuration loaded from `config/models.yaml`
- `RedFlagParser`: Validates LLM responses before voting (format, token limits, move validity)
- `MDAPHarness.first_to_ahead_by_k()`: Core voting mechanism - samples until one response leads by k_margin votes
- `MDAPHarness.execute_mdap()`: Orchestrates step execution with retry logic

**`micro_agent.py`** - Abstract base class for domain-specific agents:
- `create_initial_state()`: Initialize problem state
- `generate_step_prompt()`: Generate LLM prompt for current state
- `update_state()`: Apply step result to state
- `is_solved()`: Check termination condition

**`hanoi_solver.py`** - Towers of Hanoi implementation:
- `HanoiState`: Dataclass with pegs dict, move history
- `HanoiMDAP`: MicroAgent implementation with optimal move calculation
- Prompts follow exact format from the MAKER paper

### Key Configuration (config/models.yaml)

```yaml
model:
  name: "zai-glm-4.6"
  provider: "cerebras"
  temperature: 0.6
  disable_reasoning: true      # Cerebras-specific
  thinking_budget: 200
  max_response_length: 2048

mdap_defaults:
  k_margin: 3                  # Votes needed to win
  max_candidates: 10           # Max samples before majority vote
  max_retries: 3
```

### Environment Variables
- `MDAP_K_MARGIN`: Override k_margin from config
- `MDAP_MAX_CANDIDATES`: Override max candidates
- `MDAP_DEFAULT_MODEL`: Override default model
- `MDAP_MOCK_MODE`: Enable mock mode for testing
- `LITELLM_LOG`: Set to DEBUG for verbose LiteLLM logging

### Data Flow

1. `execute_agent_mdap()` creates initial state via agent
2. Loop until `is_solved()`:
   - `step_generator()` returns (prompt_tuple, parser)
   - `first_to_ahead_by_k()` samples LLM responses
   - `RedFlagParser` filters invalid responses
   - Voting continues until k_margin reached or max_candidates hit
   - `update_state()` applies winning response
3. Returns execution trace (list of states)

### Response Format

LLM responses must match this exact format:
```
move = [disk_id, from_peg, to_peg]
next_state = [[peg0], [peg1], [peg2]]
```

The red-flag parser validates:
- Token count under max_response_length
- Valid JSON structure
- Move format: 3-element list with valid peg indices (0-2)
- State format: list of 3 lists
- No same-peg moves (from_peg != to_peg)

## Calibration System

Calibration estimates per-step success rate (p) and calculates optimal k_margin:
1. Generate/load states from optimal Hanoi solution
2. Sample states evenly across solution space
3. For each state, compare LLM move to optimal move
4. Calculate p = successful_steps / total_valid_steps
5. Use paper's formula to derive k_min for target reliability

Calibration cache (`calibration_cache.pkl`) stores pre-generated states for 20-disk problem.

## File Structure

```
mdap/
  mdap_harness.py      # Core MAKER framework
  micro_agent.py       # Abstract agent base class
  hanoi_solver.py      # Hanoi problem implementation
  calibrate_hanoi.py   # Calibration runner
  example_hanoi.py     # Example usage
  extract_calibration_data.py  # Log analysis
  config/models.yaml   # Model and framework config
  logs/                # Runtime logs (gitignored)
```
