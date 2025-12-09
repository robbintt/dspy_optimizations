"""
MDAP (Massively Decomposed Agentic Processes) Harness
Implementation of MAKER framework: Maximal Agentic decomposition, 
first-to-ahead-by-K Error correction, and Red-flagging
"""

import asyncio
import json
import logging
import os
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
    k_margin: int = int(os.getenv("MDAP_K_MARGIN", "3"))  # First-to-ahead-by-K margin
    max_candidates: int = int(os.getenv("MDAP_MAX_CANDIDATES", "10"))  # Max candidates to sample
    temperature: float = float(os.getenv("MDAP_TEMPERATURE", "0.1"))
    max_retries: int = 3
    cost_threshold: Optional[float] = None

    # --- Model Behavior Options ---
    # Options to control model output, particularly for Cerebras/zai-glm-4.6
    # See: https://inference-docs.cerebras.ai/resources/glm-migration#7-minimize-reasoning-when-not-needed

    # Set appropriate max_completion_tokens limits. For focused responses, consider using lower values.
    # Note: LiteLLM uses the 'max_tokens' parameter, which maps to 'max_completion_tokens' in the API.
    max_tokens: int = int(os.getenv("MDAP_MAX_TOKENS", "100"))

    # Disable Reasoning with the nonstandard disable_reasoning: True parameter.
    # This is different from the 'thinking' parameter that Z.ai uses in their API.
    # Set to None to omit the parameter from the API call.
    disable_reasoning: Optional[bool] = os.getenv("MDAP_DISABLE_REASONING", "true").lower() == "true"

class RedFlagParser:
    """Red-flagging parser to filter invalid responses before voting"""
    
    @staticmethod
    def parse_move_state_flag(response: str) -> Optional[Dict[str, Any]]:
        """
        Parse and validate a move response.
        Returns None if response is flagged (invalid).
        """
        try:
            # Try to parse as JSON
            if isinstance(response, str):
                data = json.loads(response)
            else:
                data = response
            
            # Red flag 1: Check length (overly long responses)
            if len(str(data)) > 500:  # Configurable threshold
                logger.warning(f"Response too long: {len(str(data))} chars")
                return None
            
            # Red flag 2: Check required fields exist
            if not isinstance(data, dict):
                logger.warning("Response is not a dictionary")
                return None
            
            # Red flag 3: Check for empty or None critical fields
            for field in ['from_peg', 'to_peg']:
                if field not in data or data[field] is None:
                    logger.warning(f"Missing or None field: {field}")
                    return None
            
            # Red flag 4: Check valid peg values
            valid_pegs = ['A', 'B', 'C']
            if data['from_peg'] not in valid_pegs or data['to_peg'] not in valid_pegs:
                logger.warning(f"Invalid peg values: {data}")
                return None
            
            # Red flag 5: Check not moving to same peg
            if data['from_peg'] == data['to_peg']:
                logger.warning(f"Cannot move from {data['from_peg']} to same peg")
                return None
            
            return data
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            return None
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
                
                content = response.choices[0].message.content
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
            
            # Execute step with error correction
            step_result = await self.execute_step(step_prompt, response_parser)
            
            # Update state using the agent's update_state method if available
            if agent:
                current_state = agent.update_state(current_state, step_result)
            else:
                current_state = self.update_state(current_state, step_result)
            execution_trace.append(current_state)
            
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
    
    def update_state(self, current_state: Any, step_result: Any) -> Any:
        """
        Update state based on step result.
        This should be overridden by specific implementations.
        """
        raise NotImplementedError("Subclasses must implement update_state")
