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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import Counter
import litellm
from litellm import completion, acompletion

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback if python-dotenv not installed
    pass

# Configure LiteLLM logging from environment
litellm.set_verbose = os.getenv("LITELLM_LOG", "INFO").upper() == "DEBUG"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MDAPConfig:
    """Configuration for MDAP execution"""
    model: str = os.getenv("MDAP_DEFAULT_MODEL", "cerebras/zai-glm-4.6")
    k_margin: int = int(os.getenv("MDAP_K_MARGIN", "6"))  # First-to-ahead-by-K margin
    max_candidates: int = int(os.getenv("MDAP_MAX_CANDIDATES", "10"))  # Max candidates to sample
    temperature: float = float(os.getenv("MDAP_TEMPERATURE", "0.1"))
    max_retries: int = 3
    cost_threshold: Optional[float] = None

    # --- Model Behavior Options ---
    # Options to control model output, particularly for Cerebras/zai-glm-4.6
    # See: https://inference-docs.cerebras.ai/resources/glm-migration#7-minimize-reasoning-when-not-needed

    # Set appropriate max_completion_tokens limits. For focused responses, consider using lower values.
    # Note: LiteLLM uses the 'max_tokens' parameter, which maps to 'max_completion_tokens' in the API.
    max_tokens: int = int(os.getenv("MDAP_MAX_TOKENS", "1000"))
    
    # Thinking budget for models that support reasoning/thinking
    thinking_budget: int = int(os.getenv("MDAP_THINKING_BUDGET", "200"))

    # Disable Reasoning with the nonstandard disable_reasoning: True parameter.
    # This is different from the 'thinking' parameter that Z.ai uses in their API.
    # Set to None to omit the parameter from the API call.
    disable_reasoning: Optional[bool] = os.getenv("MDAP_DISABLE_REASONING", "false").lower() == "true"

class RedFlagParser:
    """Red-flagging parser to filter invalid responses before voting"""
    
    @staticmethod
    def parse_move_state_flag(response: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Parse and validate a move and next_state response.
        Returns None if response is flagged (invalid).
        """
        try:
            # Handle dict input (legacy format)
            if isinstance(response, dict):
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
            
            # Handle string input (new format)
            # Red flag 1: Check length (overly long responses)
            if len(response) > 1000:  # Increased threshold for multi-part response
                logger.warning(f"Response too long: {len(response)} chars")
                return None

            # Parse the multi-part response
            lines = response.strip().split('\n')
            move_line = None
            state_line = None

            for line in lines:
                if line.startswith("move ="):
                    move_line = line
                elif line.startswith("next_state ="):
                    state_line = line
            
            if not move_line or not state_line:
                logger.warning("Response missing 'move' or 'next_state' line")
                return None

            # Extract JSON from the lines
            try:
                move_json = move_line.split("=", 1)[1].strip()
                move_data = json.loads(move_json)
            except (json.JSONDecodeError, IndexError) as e:
                logger.warning(f"Failed to parse move JSON: {e}")
                return None

            try:
                state_json = state_line.split("=", 1)[1].strip()
                predicted_state = json.loads(state_json)
            except (json.JSONDecodeError, IndexError) as e:
                logger.warning(f"Failed to parse next_state JSON: {e}")
                return None

            # Red flag 2: Check move structure
            if not isinstance(move_data, dict) or 'from_peg' not in move_data or 'to_peg' not in move_data:
                logger.warning("Move is not a valid dictionary")
                return None
            
            # Red flag 3: Check for empty or None critical fields
            if move_data['from_peg'] is None or move_data['to_peg'] is None:
                logger.warning("Move contains None fields")
                return None
            
            # Red flag 4: Check valid peg values
            valid_pegs = ['A', 'B', 'C']
            if move_data['from_peg'] not in valid_pegs or move_data['to_peg'] not in valid_pegs:
                logger.warning(f"Invalid peg values: {move_data}")
                return None
            
            # Red flag 5: Check not moving to same peg
            if move_data['from_peg'] == move_data['to_peg']:
                logger.warning(f"Cannot move from {move_data['from_peg']} to same peg")
                return None

            # Red flag 6: Check predicted state structure
            if not isinstance(predicted_state, dict) or 'pegs' not in predicted_state:
                logger.warning("Predicted state is not a valid dictionary")
                return None

            # Return the move and the predicted state for later validation
            return {
                "move": move_data,
                "predicted_state": predicted_state
            }
            
        except Exception as e:
            logger.warning(f"Validation error: {e}")
            return None

class MDAPHarness:
    """Main MDAP harness implementing MAKER framework"""
    
    def __init__(self, config: MDAPConfig):
        self.config = config
        self.red_flag_parser = RedFlagParser()
        self.total_cost = 0.0
        
    async def first_to_ahead_by_k(self, 
                                 prompt: str, 
                                 response_parser: Callable[[str], Any]) -> Any:
        """
        First-to-ahead-by-K voting mechanism
        Samples candidates until one leads by K votes
        """
        votes = Counter()
        candidates = []
        
        async def get_candidate():
            """Get a single candidate response"""
            try:
                # Build completion parameters, including optional model-specific ones
                completion_params = {
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                }

                # Add disable_reasoning if specified in the config
                if self.config.disable_reasoning is not None:
                    completion_params["disable_reasoning"] = self.config.disable_reasoning
                
                response = await acompletion(**completion_params)
                
                message = response.choices[0].message
                content = message.content
                
                # Check if we have reasoning content but no final content
                if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
                    logger.warning(f"LLM returned reasoning but no final content. This often means max_tokens was too small.")
                    logger.warning(f"Reasoning was: {message.reasoning_content[:200]}...")
                    return None
                
                if content is None:
                    logger.warning(f"LLM returned None content. Full response: {response}")
                    return None
                return content.strip()
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return None
        
        # Sample candidates until we have a winner or hit max limit
        attempts = 0
        while len(candidates) < self.config.max_candidates and attempts < self.config.max_candidates:
            attempts += 1
            # Get new candidate
            raw_response = await get_candidate()
            if raw_response is None:
                continue

            # Add this line to log the raw response
            logger.info(f"LLM Raw Response: {raw_response}")
                
            # Apply red-flagging
            parsed_response = response_parser(raw_response)
            if parsed_response is None:
                logger.info("Response red-flagged, continuing...")
                continue
            
            # Convert to hashable for voting
            response_key = json.dumps(parsed_response, sort_keys=True)
            votes[response_key] += 1
            candidates.append(parsed_response)
            
            # Check if we have a winner
            if votes[response_key] >= self.config.k_margin:
                logger.info(f"Winner found with {votes[response_key]} votes")
                return parsed_response
            
            # Check if any candidate leads by K
            sorted_votes = votes.most_common()
            if len(sorted_votes) >= 2:
                leader_votes = sorted_votes[0][1]
                runner_up_votes = sorted_votes[1][1]
                if leader_votes - runner_up_votes >= self.config.k_margin:
                    winner_key = sorted_votes[0][0]
                    winner = json.loads(winner_key)
                    logger.info(f"Winner leads by {leader_votes - runner_up_votes} votes")
                    return winner
        
        # If no clear winner, return majority vote
        if votes:
            winner_key = votes.most_common(1)[0][0]
            winner = json.loads(winner_key)
            logger.warning(f"No clear winner, returning majority vote")
            return winner
        
        raise Exception("No valid candidates found")
    
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
        current_state = initial_state
        execution_trace = [current_state]
        step_count = 0
        
        while not termination_check(current_state):
            step_count += 1
            logger.info(f"Executing step {step_count}")
            
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
                        current_state = agent.update_state(current_state, step_result)
                    else:
                        current_state = self.update_state(current_state, step_result)
                    
                    # If both succeed, break the retry loop and continue
                    break

                except Exception as e:
                    last_exception = e
                    logger.error(f"Step execution attempt {attempt + 1} failed: {e}")
                    # If this is not the last attempt, modify the prompt to include the error
                    if attempt < self.config.max_retries - 1:
                        error_context = f"\n\nERROR: Your previous move was invalid!\n{str(e)}\n\nREMEMBER: Larger disks CANNOT go on smaller disks.\nIf moving disk X to peg Y, check that peg Y's top disk (if any) is larger than X.\n\nPlease analyze the current state carefully and choose a VALID move:"
                        current_step_prompt = step_prompt + error_context
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
                # We only need one successful candidate to check for validity
                step_result = await self.first_to_ahead_by_k(step_prompt, response_parser)
                logger.info(f"Step {actual_steps_attempted} LLM call successful")
                # The update_state method itself acts as the validation
                new_state = agent.update_state(current_state, step_result)
                logger.info(f"Step {actual_steps_attempted} state update successful")
                successful_steps += 1
                current_state = new_state
                logger.info(f"New state: {current_state.to_dict() if hasattr(current_state, 'to_dict') else current_state}")
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
