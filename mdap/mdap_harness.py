"""
MDAP (Massively Decomposed Agentic Processes) Harness
Implementation of MAKER framework: Maximal Agentic decomposition, 
first-to-ahead-by-K Error correction, and Red-flagging
"""

import asyncio
import json
import logging
import os
import math
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
import yaml
import litellm
from litellm import completion, acompletion

# Setup logging to file with timestamps
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOGS_DIR, f"mdap_harness_{timestamp}.log")

# Configure file handler for MDAP logs
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to root logger
logging.getLogger().addHandler(file_handler)

# Also add console handler to tee output to terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback if python-dotenv not installed
    pass

# Configure LiteLLM logging from environment
litellm.set_verbose = os.getenv("LITELLM_LOG", "INFO").upper() == "DEBUG"

# Don't call basicConfig here since handlers are already configured above
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MDAPConfig:
    """Configuration for MDAP execution"""
    
    def __init__(self, **kwargs):
        # Load configuration from YAML
        config_file = Path(__file__).parent / "config" / "models.yaml"
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        model_config = config['model']
        mdap_defaults = config['mdap_defaults']
        
        # Model settings (allow override via kwargs)
        # If model doesn't include provider, add it from config
        model = kwargs.get('model', model_config['name'])
        if '/' not in model:
            self.model = f"{model_config['provider']}/{model}"
        else:
            self.model = model
        self.temperature = kwargs.get('temperature', model_config.get('temperature', 0.6))
        self.max_tokens = kwargs.get('max_tokens', model_config.get('max_tokens', 2048))
        self.cost_per_input_token = model_config.get('cost_per_input_token', 0.00015)
        self.cost_per_output_token = model_config.get('cost_per_output_token', 0.0006)
        self.max_response_length = model_config.get('max_response_length', 750)
        
        # Cerebras-specific options
        self.disable_reasoning = model_config.get('disable_reasoning', None)
        self.reasoning_effort = model_config.get('reasoning_effort', None)
        self.thinking_budget = model_config.get('thinking_budget', 200)
        
        # MDAP framework settings (allow override via kwargs, then env vars)
        self.k_margin = kwargs.get('k_margin', int(os.getenv("MDAP_K_MARGIN", str(mdap_defaults['k_margin']))))
        self.max_candidates = kwargs.get('max_candidates', int(os.getenv("MDAP_MAX_CANDIDATES", str(mdap_defaults['max_candidates']))))
        self.max_retries = mdap_defaults['max_retries']
        self.cost_threshold = mdap_defaults['cost_threshold']
        
        # Other settings
        self.mock_mode = os.getenv("MDAP_MOCK_MODE", "false").lower() == "true"

class RedFlagParser:
    """Red-flagging parser to filter invalid responses before voting"""
    
    def __init__(self, config: MDAPConfig):
        self.config = config
    
    def parse_move_state_flag(self, response: Union[str, Dict[str, Any]], usage: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        Parse and validate a move and next_state response using the paper's format.
        Non-repairing extractor: If the extractor fails, we discard per red flagging.
        Returns None if response is flagged (invalid).
        
        Args:
            response: The response text to parse
            usage: Optional usage object from the API response containing token counts
        """
        try:
            # Handle string input (paper's format)
            if isinstance(response, str):
                # Red flag 1: Check token limit (overly long responses)
                if usage and hasattr(usage, 'completion_tokens'):
                    token_count = usage.completion_tokens
                    if token_count > self.config.max_response_length:
                        logger.warning(f"RED FLAG: Response too long: {token_count} tokens > {self.config.max_response_length}")
                        return None
                else:
                    # Fallback to character count if no usage info available
                    if len(response) > self.config.max_response_length * 4:  # Rough estimate: 4 chars per token
                        logger.warning(f"RED FLAG: Response too long (fallback): {len(response)} chars")
                        return None

                # Parse the multi-part response
                lines = response.strip().split('\n')
                move_line = None
                state_line = None

                for line in lines:
                    # Strict matching for move line - must start exactly with "move ="
                    if line.strip().startswith("move ="):
                        move_line = line
                    # Strict matching for state line - must start exactly with "next_state ="
                    elif line.strip().startswith("next_state ="):
                        state_line = line
                
                # If we didn't find the required lines with strict matching, try to extract from code blocks
                if not move_line or not state_line:
                    # Look for code blocks that might contain the move and state
                    import re
                    
                    # Try to find move in a code block
                    if not move_line:
                        move_match = re.search(r'```(?:python)?\s*move\s*=\s*(\[[^\]]+\])\s*```', response, re.IGNORECASE | re.DOTALL)
                        if move_match:
                            move_line = f"move = {move_match.group(1)}"
                    
                    # Try to find next_state in a code block
                    if not state_line:
                        state_match = re.search(r'```(?:python)?\s*next_state\s*=\s*(\[[^\]]+\]|\{[^}]+\})\s*```', response, re.IGNORECASE | re.DOTALL)
                        if state_match:
                            state_line = f"next_state = {state_match.group(1)}"
                    
                    # If still not found, try to find them without code blocks but with proper formatting
                    if not move_line:
                        move_match = re.search(r'move\s*=\s*(\[[^\]]+\])', response, re.IGNORECASE)
                        if move_match:
                            move_line = f"move = {move_match.group(1)}"
                    
                    if not state_line:
                        state_match = re.search(r'next_state\s*=\s*(\[[^\]]+\]|\{[^}]+\})', response, re.IGNORECASE)
                        if state_match:
                            state_line = f"next_state = {state_match.group(1)}"
                
                if not move_line or not state_line:
                    logger.warning(f"RED FLAG: Response missing required fields - move_line={bool(move_line)}, state_line={bool(state_line)}")
                    return None

                # Extract JSON from the lines
                try:
                    move_json = move_line.split("=", 1)[1].strip()
                    # Allow markdown code blocks by stripping them if present
                    if move_json.startswith('```') and move_json.endswith('```'):
                        move_json = move_json[3:-3].strip()
                    # The value is a JSON array, not an object
                    move_data = json.loads(move_json)
                except (json.JSONDecodeError, IndexError) as e:
                    logger.warning(f"RED FLAG: Failed to parse move JSON: {e}")
                    logger.warning(f"Move line was: {move_line}")
                    return None

                try:
                    state_json = state_line.split("=", 1)[1].strip()
                    # Allow markdown code blocks by stripping them if present
                    if state_json.startswith('```') and state_json.endswith('```'):
                        state_json = state_json[3:-3].strip()
                    # The value is a JSON array, not an object
                    predicted_state = json.loads(state_json)
                except (json.JSONDecodeError, IndexError) as e:
                    logger.warning(f"RED FLAG: Failed to parse next_state JSON: {e}")
                    logger.warning(f"State line was: {state_line}")
                    return None

                # Red flag 2: Check move structure (paper format: [disk_id, from_peg, to_peg])
                if not isinstance(move_data, list) or len(move_data) != 3:
                    logger.warning("Move is not a valid list of 3 elements")
                    return None
                
                disk_id, from_peg, to_peg = move_data
                
                # Red flag 3: Check for empty or None critical fields
                if None in move_data:
                    logger.warning("Move contains None fields")
                    return None
                
                # Red flag 4: Check valid values
                if not isinstance(disk_id, int) or disk_id < 1:
                    logger.warning(f"Invalid disk_id: {disk_id}")
                    return None
                if not isinstance(from_peg, int) or from_peg not in [0, 1, 2]:
                    logger.warning(f"Invalid from_peg: {from_peg}")
                    return None
                if not isinstance(to_peg, int) or to_peg not in [0, 1, 2]:
                    logger.warning(f"Invalid to_peg: {to_peg}")
                    return None
                
                # Red flag 5: Check not moving to same peg
                if from_peg == to_peg:
                    logger.warning(f"Cannot move from {from_peg} to same peg")
                    return None

                # Red flag 6: Check predicted state structure
                # Handle both {"pegs": [...]} format and direct [...] format
                if isinstance(predicted_state, dict) and 'pegs' in predicted_state:
                    pegs = predicted_state['pegs']
                elif isinstance(predicted_state, list):
                    # LLM returned pegs list directly
                    pegs = predicted_state
                    # Convert to expected dict format for consistency
                    predicted_state = {'pegs': pegs}
                else:
                    logger.warning(f"Predicted state is not a valid dictionary or list: {type(predicted_state)}")
                    return None
                
                # Red flag 7: Check pegs structure
                if not isinstance(pegs, list) or len(pegs) != 3:
                    logger.warning("Pegs must be a list of 3 lists")
                    return None
                if not all(isinstance(peg, list) for peg in pegs):
                    logger.warning("Each peg must be a list")
                    return None

                # Ensure predicted_state has move_count for compatibility
                if isinstance(predicted_state, dict) and 'move_count' not in predicted_state:
                    # If move_count is missing, we'll add it later in update_state
                    pass
                
                # Return the move and the predicted state for later validation
                return {
                    "move": move_data,
                    "predicted_state": predicted_state
                }
            
            # Handle dict input (legacy format) - still support for backward compatibility
            elif isinstance(response, dict):
                # Red flag 2: Check move structure
                if not isinstance(response, dict) or 'from_peg' not in response or 'to_peg' not in response:
                    logger.warning("Dict response is not a valid move dictionary")
                    return None
                
                # Red flag 3: Check for empty or None critical fields
                if response['from_peg'] is None or response['to_peg'] is None:
                    logger.warning("Dict response contains None fields")
                    return None
                
                # Red flag 4: Check valid peg values
                valid_pegs = ['A', 'B', 'C']
                if response['from_peg'] not in valid_pegs or response['to_peg'] not in valid_pegs:
                    logger.warning(f"Dict response has invalid peg values: {response}")
                    return None
                
                # Red flag 5: Check not moving to same peg
                if response['from_peg'] == response['to_peg']:
                    logger.warning(f"Dict response cannot move from {response['from_peg']} to same peg")
                    return None
                
                # Return in the expected format
                return {
                    "move": response,
                    "predicted_state": None  # Not available in dict format
                }
            
        except Exception as e:
            logger.warning(f"Validation error: {e}")
            return None

class MDAPHarness:
    """Main MDAP harness implementing MAKER framework"""
    
    def __init__(self, config: MDAPConfig):
        self.config = config
        self.red_flag_parser = RedFlagParser(config)
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_api_calls = 0
        self.temperature_first_vote = 0.6
        
    async def first_to_ahead_by_k(self, 
                                 prompt: str, 
                                 response_parser: Callable[[str], Any]) -> Any:
        """
        First-to-ahead-by-K voting mechanism
        Samples candidates until one leads by K votes
        """
        logger.info(f"Starting first-to-ahead-by-K with k_margin={self.config.k_margin}, max_candidates={self.config.max_candidates}")
        logger.info(f"Prompt type: {type(prompt)}, length: {len(prompt) if prompt else 'None'}")
        logger.info(f"Response parser type: {type(response_parser)}")
        votes = Counter()
        candidates = []
        first_vote = True  # Track if this is the first vote
        
        async def get_candidate():
            """Get a single candidate response with non-repairing extractor"""
            nonlocal first_vote
            try:
                # Check if we're in mock mode
                if self.config.mock_mode:
                    # Return a mock valid response for testing
                    mock_response = """move = [1, 0, 2]
next_state = {"pegs": [[2, 3], [], [1]]}"""
                    logger.warning("MOCK MODE ENABLED - returning mock response instead of calling LLM")
                    logger.info(f"Mock response: {mock_response}")
                    try:
                        return response_parser(mock_response.strip())
                    except Exception as e:
                        logger.error(f"Mock response parser failed: {e}")
                        return None
                
                # Use temperature=0 for first vote, 0.1 for subsequent votes (per paper)
                temperature = self.temperature_first_vote if first_vote else self.config.temperature
                if first_vote:
                    logger.info(f"Using temperature={self.temperature_first_vote} for first vote to ensure best guess")
                    first_vote = False
                logger.info(f"Making LLM call with temperature={temperature}")
                
                # Build messages for chat completion
                system_prompt, user_prompt = prompt
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # Build completion parameters, including optional model-specific ones
                completion_params = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": self.config.max_tokens,
                }

                # Add disable_reasoning if specified in the config and not None
                if self.config.disable_reasoning is not None:
                    completion_params["disable_reasoning"] = self.config.disable_reasoning
                
                # Add reasoning_effort if specified in the config and not None
                if hasattr(self.config, 'reasoning_effort') and self.config.reasoning_effort is not None:
                    completion_params["reasoning_effort"] = self.config.reasoning_effort
                
                # Time the API call
                api_start = time.time()
                response = await acompletion(**completion_params)
                api_time = time.time() - api_start
                
                message = response.choices[0].message
                content = message.content
                
                # Extract token usage
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                
                # Calculate cost
                call_cost = (input_tokens * self.config.cost_per_input_token + 
                           output_tokens * self.config.cost_per_output_token)
                
                # Update cumulative statistics
                self.total_cost += call_cost
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_api_calls += 1
                
                # Log API call details
                logger.info(f"API Call: model={self.config.model}, "
                           f"in_tokens={input_tokens}, out_tokens={output_tokens}, "
                           f"cost=${call_cost:.6f}, temp={completion_params.get('temperature')}, "
                           f"time={api_time:.2f}s, length={len(content) if content else 0} chars, "
                           f"cumulative: total_cost=${self.total_cost:.4f}, "
                           f"total_calls={self.total_api_calls}")
                
                # Check if we have reasoning content but no final content
                if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                    logger.warning(f"RED FLAG: LLM returned reasoning but no final content. This often means max_tokens was too small.")
                    logger.warning(f"Reasoning was: {message.reasoning_content[:200]}...")
                    return None
                
                if content is None:
                    logger.warning(f"RED FLAG: LLM returned None content. Full response: {response}")
                    return None
                
                # Log raw response for debugging
                logger.info(f"RAW LLM RESPONSE (length={len(content)}):")
                logger.info("-" * 80)
                logger.info(content)
                logger.info("-" * 80)
                
                # Check if content is empty or None
                if not content or content.strip() == "":
                    logger.error("RED FLAG: LLM returned empty content")
                    return None
                
                # Apply red flagging (non-repairing extractor)
                try:
                    # Pass usage info to the parser if it accepts it
                    if hasattr(response_parser, '__code__') and response_parser.__code__.co_argcount > 1:
                        parsed_response = response_parser(content.strip(), usage)
                    else:
                        parsed_response = response_parser(content.strip())
                    if parsed_response is None:
                        logger.warning(f"RED FLAG: Response discarded by red-flag parser")
                        return None
                except Exception as e:
                    logger.error(f"RED FLAG: Response parser threw exception: {e}")
                    logger.error(f"Response content was: {content[:200]}...")
                    return None
                
                logger.info(f"Response passed red-flagging: {parsed_response}")
                return parsed_response
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return None
        
        # Sample candidates until we have a winner or hit max limit
        attempts = 0
        while len(candidates) < self.config.max_candidates and attempts < self.config.max_candidates:
            attempts += 1
            logger.info(f"Attempting to get candidate {attempts}/{self.config.max_candidates}")
            
            # Check if prompt is valid
            if not prompt or not isinstance(prompt, tuple) or len(prompt) != 2:
                logger.error(f"Invalid prompt: {prompt}")
                return None
            
            # Get new candidate (already parsed and red-flagged)
            parsed_response = await get_candidate()
            if parsed_response is None:
                logger.info(f"Candidate {attempts} was red-flagged and discarded")
                continue

            # Add this line to log the parsed response
            logger.info(f"LLM Parsed Response: {parsed_response}")
            
            # Convert to hashable for voting
            response_key = json.dumps(parsed_response, sort_keys=True)
            votes[response_key] += 1
            candidates.append(parsed_response)
            
            # Check if we have a winner
            if votes[response_key] >= self.config.k_margin:
                # Calculate vote margin
                sorted_votes = votes.most_common()
                runner_up_votes = sorted_votes[1][1] if len(sorted_votes) > 1 else 0
                vote_margin = votes[response_key] - runner_up_votes
                
                logger.info(f"Voting Result: winner='{str(parsed_response)[:50]}...', "
                           f"votes={votes[response_key]}/{attempts}, "
                           f"margin={vote_margin}, unique_responses={len(votes)}, "
                           f"reached_k_margin=True")
                logger.info(f"Winner found with {votes[response_key]} votes (reached k_margin)")
                logger.info(f"Winning response: {parsed_response}")
                return parsed_response
            
            # Check if any candidate leads by K
            sorted_votes = votes.most_common()
            if len(sorted_votes) >= 2:
                leader_votes = sorted_votes[0][1]
                runner_up_votes = sorted_votes[1][1]
                if leader_votes - runner_up_votes >= self.config.k_margin:
                    winner_key = sorted_votes[0][0]
                    winner = json.loads(winner_key)
                    vote_margin = leader_votes - runner_up_votes
                    
                    logger.info(f"Voting Result: winner='{str(winner)[:50]}...', "
                               f"votes={leader_votes}/{attempts}, "
                               f"margin={vote_margin}, unique_responses={len(votes)}, "
                               f"reached_k_margin=True")
                    logger.info(f"Winner leads by {vote_margin} votes")
                    return winner
        
        # If no clear winner, return majority vote
        if votes:
            winner_key = votes.most_common(1)[0][0]
            winner = json.loads(winner_key)
            winner_votes = votes.most_common(1)[0][1]
            
            logger.info(f"Voting Result: winner='{str(winner)[:50]}...', "
                       f"votes={winner_votes}/{attempts}, "
                       f"margin=0, unique_responses={len(votes)}, "
                       f"reached_k_margin=False (majority vote)")
            logger.warning(f"No clear winner, returning majority vote")
            return winner
        
        logger.warning(f"No valid candidates found after {attempts} attempts (all were red-flagged)")
        return None
    
    async def execute_step(self, 
                          step_prompt: str, 
                          response_parser: Callable[[str], Any]) -> Any:
        """
        Execute a single atomic step with error correction
        """
        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                result = await self.first_to_ahead_by_k(step_prompt, response_parser)
                return result
            except Exception as e:
                last_exception = e
                logger.error(f"Step execution attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    break
        
        raise Exception(f"Step execution failed after {self.config.max_retries} attempts")
    
    async def execute_mdap(self, 
                          initial_state: Any,
                          step_generator: Callable[[Any], Tuple[str, Callable[[str], Any]]],
                          termination_check: Callable[[Any], bool],
                          agent: 'MicroAgent' = None) -> List[Any]:
        """
        Execute a complete MDAP process
        """
        logger.info("Starting MDAP execution")
        logger.info(f"Initial state: {initial_state}")
        current_state = initial_state
        execution_trace = [current_state]
        step_count = 0
        
        while not termination_check(current_state):
            step_count += 1
            logger.info(f"Executing step {step_count}")
            logger.info(f"Current state before step: {current_state}")
            
            # Generate prompt and parser for current step
            step_prompt, response_parser = step_generator(current_state)
            
            # Execute step and update state with error correction
            last_exception = None
            current_step_prompt = step_prompt
            for attempt in range(self.config.max_retries):
                try:
                    # Execute step to get a result from the LLM
                    step_result = await self.execute_step(current_step_prompt, response_parser)
                    
                    # Update state using the agent's update_state method if available
                    if agent:
                        logger.info(f"Updating state using agent method with step result: {step_result}")
                        current_state = agent.update_state(current_state, step_result)
                    else:
                        logger.info(f"Updating state using harness method with step result: {step_result}")
                        current_state = self.update_state(current_state, step_result)
                    
                    logger.info(f"State updated successfully: {current_state}")
                    # If both succeed, break the retry loop and continue
                    break

                except Exception as e:
                    last_exception = e
                    logger.error(f"Step execution attempt {attempt + 1} failed: {e}")
                    # If this is not the last attempt, modify the prompt to include the error
                    if attempt < self.config.max_retries - 1:
                        error_context = f"\n\nERROR: Your previous move was invalid!\n{str(e)}\n\nREMEMBER: Larger disks CANNOT go on smaller disks.\nIf moving disk X to peg Y, check that peg Y's top disk (if any) is larger than X.\n\nPlease analyze the current state carefully and choose a VALID move:"
                        # step_prompt is a tuple (system_prompt, user_prompt)
                        # We need to append the error_context to the user_prompt string
                        system_prompt, user_prompt = step_prompt
                        current_step_prompt = (system_prompt, user_prompt + error_context)
                    else:
                        # If max retries reached, re-raise the exception to stop execution
                        raise Exception(f"Step execution failed after {self.config.max_retries} attempts") from e
            
            execution_trace.append(current_state)
            
            # Check if solved AFTER updating state and BEFORE next iteration
            if termination_check(current_state):
                logger.info(f"Goal reached after {step_count} steps")
                break
            
            # Check cost threshold
            if self.config.cost_threshold and self.total_cost > self.config.cost_threshold:
                logger.warning(f"Cost threshold exceeded: ${self.total_cost:.4f}")
                break
        
        logger.info(f"MDAP execution completed in {step_count} steps")
        return execution_trace
    
    async def execute_agent_mdap(self, agent: 'MicroAgent', *args, **kwargs) -> List[Any]:
        """
        Execute MDAP using a micro agent
        
        Args:
            agent: The micro agent to execute
            *args, **kwargs: Arguments to pass to agent.create_initial_state()
            
        Returns:
            List of states representing the execution trace
        """
        initial_state = agent.create_initial_state(*args, **kwargs)
        
        return await self.execute_mdap(
            initial_state=initial_state,
            step_generator=agent.step_generator,
            termination_check=agent.is_solved,
            agent=agent
        )
    
    async def estimate_per_step_success_rate(self, agent: 'MicroAgent', num_disks: int, sample_steps: int = 10) -> float:
        """
        Estimates the per-step success rate (p) for a given model and agent.
        Runs the agent for a small number of steps and counts successes.
        Now checks against known optimal moves instead of just validity.
        """
        logger.info(f"Estimating per-step success rate for {self.config.model} on {sample_steps} steps...")
        initial_state = agent.create_initial_state(num_disks)
        logger.info(f"Initial state: {initial_state.to_dict() if hasattr(initial_state, 'to_dict') else initial_state}")
        current_state = initial_state
        successful_steps = 0
        actual_steps_attempted = 0
        
        for i in range(sample_steps):
            logger.info(f"Estimation loop iteration {i+1}/{sample_steps}")
            if agent.is_solved(current_state):
                logger.info(f"Agent solved the problem after {actual_steps_attempted} steps")
                break
            
            step_prompt, response_parser = agent.step_generator(current_state)
            actual_steps_attempted += 1
            logger.info(f"Attempting step {actual_steps_attempted}")
            
            try:
                # Get the optimal move for this state
                optimal_move = agent.get_optimal_move(current_state)
                logger.info(f"Optimal move for step {actual_steps_attempted}: {optimal_move}")
                
                # We only need one successful candidate to check against optimal
                step_result = await self.first_to_ahead_by_k(step_prompt, response_parser)
                
                # If step_result is None, it means all candidates were red-flagged
                if step_result is None:
                    logger.warning(f"Step {actual_steps_attempted}: All candidates were red-flagged and discarded ✗")
                    break
                
                logger.info(f"Step {actual_steps_attempted} LLM call successful")
                
                # Check if the LLM's move matches the optimal move
                llm_move = step_result.get("move", [])
                logger.info(f"LLM move: {llm_move}")
                
                if llm_move == optimal_move:
                    logger.info(f"Step {actual_steps_attempted}: LLM move matches optimal move ✓")
                    successful_steps += 1
                    # Update state with the correct move
                    new_state = agent.update_state(current_state, step_result)
                    current_state = new_state
                    logger.info(f"New state: {current_state.to_dict() if hasattr(current_state, 'to_dict') else current_state}")
                else:
                    logger.warning(f"Step {actual_steps_attempted}: LLM move {llm_move} != optimal move {optimal_move} ✗")
                    # Still update state to continue calibration, but don't count as success
                    try:
                        new_state = agent.update_state(current_state, step_result)
                        current_state = new_state
                    except Exception as e:
                        logger.error(f"Failed to update state with suboptimal move: {e}")
                        break
                        
            except Exception as e:
                # A failure here means the step was unsuccessful
                logger.error(f"Estimation step {actual_steps_attempted} failed: {e}")
                break
        
        # If we didn't attempt any steps (e.g., already solved), return 1.0
        if actual_steps_attempted == 0:
            p_estimate = 1.0
        else:
            p_estimate = successful_steps / actual_steps_attempted
        
        logger.info(f"Final count: successful_steps={successful_steps}, actual_steps_attempted={actual_steps_attempted}")
        logger.info(f"Estimated per-step success rate (p): {p_estimate:.4f} ({successful_steps}/{actual_steps_attempted} steps)")
        return p_estimate

    async def estimate_per_step_success_rate_from_states(self, agent: 'MicroAgent', states: list) -> float:
        """
        Estimate the per-step success rate using pre-generated states.
        Checks against known optimal moves instead of just validity.
        """
        logger.info(f"Estimating per-step success rate for {self.config.model} on {len(states)} pre-generated states...")
        successful_steps = 0
        
        try:
            for i, state in enumerate(states):
                logger.info(f"Testing pre-generated state {i+1}/{len(states)}")
                logger.info(f"State type: {type(state)}")
                logger.info(f"State: {state}")
                
                try:
                    # Get the optimal move for this state
                    optimal_move = agent.get_optimal_move(state)
                    logger.info(f"Optimal move for state {i+1}: {optimal_move}")
                    
                    # Get step prompt and parser
                    step_prompt, response_parser = agent.step_generator(state)
                    logger.info(f"Step prompt type: {type(step_prompt)}")
                    logger.info(f"Response parser type: {type(response_parser)}")
                    
                    # We only need one successful candidate to check against optimal
                    logger.info(f"Step prompt for state {i+1}: {step_prompt[:200]}...")
                    logger.info(f"About to call first_to_ahead_by_k...")
                    step_result = await self.first_to_ahead_by_k(step_prompt, response_parser)
                    
                    # If step_result is None, it means all candidates were red-flagged
                    if step_result is None:
                        logger.warning(f"State {i+1}: All candidates were red-flagged and discarded ✗")
                        continue
                    
                    logger.info(f"State {i+1} LLM call successful")
                
                    # Check if the LLM's move matches the optimal move
                    llm_move = step_result.get("move", [])
                    logger.info(f"LLM move: {llm_move}")
                    
                    if llm_move == optimal_move:
                        logger.info(f"State {i+1}: LLM move matches optimal move ✓")
                        successful_steps += 1
                    else:
                        logger.warning(f"State {i+1}: LLM move {llm_move} != optimal move {optimal_move} ✗")
                        
                except Exception as e:
                    # A failure here means the step was unsuccessful
                    logger.error(f"Estimation for state {i+1} failed: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
        
        except Exception as e:
            logger.error(f"Unexpected error in estimate_per_step_success_rate_from_states: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        p_estimate = successful_steps / len(states) if states else 0.0
        
        logger.info(f"Final count: successful_steps={successful_steps}, total_states={len(states)}")
        logger.info(f"Estimated per-step success rate (p): {p_estimate:.4f} ({successful_steps}/{len(states)} steps)")
        return p_estimate

    def calculate_k_min(self, p: float, num_disks: int, target_reliability: float = 0.95) -> int:
        """
        Calculates the minimal k_margin based on the paper's formula.
        k_min = ceil( ln(t^(-1/s) - 1) / ln((1-p)/p) )
        where s = 2^D - 1 for Towers of Hanoi.
        """
        if p <= 0.5:
            logger.warning("Per-step success rate p is <= 0.5, voting may not converge. Using a high default k.")
            return 20 # A high default to signal failure

        total_steps = (2 ** num_disks) - 1
        if total_steps == 0: return 1

        # Handle the special case where p=1 (perfect success rate)
        if p >= 0.9999:  # Account for floating point precision
            # With perfect success rate, we only need k=1 for any reliability
            if num_disks < 10:
                logger.warning(f"⚠️  Perfect success rate (p={p:.4f}) on only {num_disks} disks. ")
                logger.warning(f"   This may not be representative of performance on larger problems.")
                logger.warning(f"   Consider re-running calibration with more disks (e.g., --sample_steps 10 or 20)")
                logger.warning(f"   for a more reliable k_margin estimate.")
            logger.info(f"Perfect success rate (p={p:.4f}), using k_min=1")
            return 1

        # From the paper: k_min = ceil( ln(t^(-m/s) - 1) / ln((1-p)/p) )
        # For MAD, m=1, so t^(-1/s)
        try:
            numerator = math.log((target_reliability ** (-1 / total_steps)) - 1)
            denominator = math.log((1 - p) / p)
            k_min_float = numerator / denominator
            k_min = math.ceil(k_min_float)
        except (ValueError, ZeroDivisionError):
            logger.error("Could not calculate k_min, likely due to p=1 or p=0.5. Using default.")
            k_min = 3 # Fallback

        logger.info(f"Calculated k_min: {k_min} for p={p:.4f}, D={num_disks}, t={target_reliability}")
        return k_min

    def update_state(self, current_state: Any, step_result: Any) -> Any:
        """
        Update state based on step result.
        This should be overridden by specific implementations.
        """
        raise NotImplementedError("Subclasses must implement update_state")
