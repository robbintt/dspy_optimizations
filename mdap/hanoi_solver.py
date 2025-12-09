"""
Towers of Hanoi solver using MDAP harness
Demonstrates MAKER framework on a classic recursive problem
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Tuple, Callable, Any
from dataclasses import dataclass
import copy
from .micro_agent import MicroAgent
from .mdap_harness import MDAPHarness, MDAPConfig, RedFlagParser

# Setup logging to file with timestamps
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOGS_DIR, f"hanoi_solver_{timestamp}.log")

# Configure file handler for Hanoi solver logs
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Also add console handler to tee output to terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Create a specific logger for hanoi_solver to avoid conflicts
hanoi_logger = logging.getLogger('hanoi_solver')
hanoi_logger.setLevel(logging.INFO)
hanoi_logger.addHandler(file_handler)
hanoi_logger.addHandler(console_handler)

logger = logging.getLogger('hanoi_solver')

# Import the system prompt and user template from the paper
# Note: We had to add token limit requirement to SYSTEM_PROMPT for GLM 4.6
SYSTEM_PROMPT = """
You are a helpful assistant. Solve this puzzle for me.
There are three pegs and n disks of different sizes stacked on the first peg. The disks are
numbered from 1 (smallest) to n (largest). Disk moves in this puzzle should follow:
1. Only one disk can be moved at a time.
2. Each move consists of taking the upper disk from one stack and placing it on top of
another stack.
3. A larger disk may not be placed on top of a smaller disk.
The goal is to move the entire stack to the third peg.
Example: With 3 disks numbered 1 (smallest), 2, and 3 (largest), the initial state is [[3, 2,
1], [], []], and a solution might be:
moves = [[1, 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2], [1, 1, 0], [2, 1, 2], [1, 0, 2]]
This means: Move disk 1 from peg 0 to peg 2, then move disk 2 from peg 0 to peg 1, and so on.
Requirements:
- The positions are 0-indexed (the leftmost peg is 0).
- Ensure your answer includes a single next move in this EXACT FORMAT:
```move = [disk id, from peg, to peg]```
- Ensure your answer includes the next state resulting from applying the move to the current
state in this EXACT FORMAT:
```next_state = [[...], [...], [...]]```
The response must be under {token_limit} tokens.
"""

USER_TEMPLATE = """
Rules:
- Only one disk can be moved at a time.
- Only the top disk from any stack can be moved.
- A larger disk may not be placed on top of a smaller disk.
For all moves, follow the standard Tower of Hanoi procedure:
If the previous move did not move disk 1, move disk 1 clockwise one peg (0 -> 1 -> 2 -> 0).
If the previous move did move disk 1, make the only legal move that does not involve moving
disk1.
Use these clear steps to find the next move given the previous move and current state.

Previous move: {previous_move}
Current State: {current_state}
Based on the previous move and current state, find the single next move that follows the
procedure and the resulting next state.
"""

@dataclass
class HanoiState:
    """State representation for Towers of Hanoi"""
    pegs: dict  # {'A': [largest...smallest], 'B': [...], 'C': [...]}
    num_disks: int
    move_count: int = 0
    move_history: List[dict] = None  # Track move history
    previous_move: List[int] = None  # Just the previous move [disk_id, from_peg, to_peg]
    
    def to_dict(self) -> dict:
        return {
            'pegs': self.pegs,
            'num_disks': self.num_disks,
            'move_count': self.move_count
        }
    
    def copy(self):
        return HanoiState(
            pegs=copy.deepcopy(self.pegs),
            num_disks=self.num_disks,
            move_count=self.move_count,
            move_history=copy.deepcopy(self.move_history) if self.move_history else [],
            previous_move=copy.deepcopy(self.previous_move) if self.previous_move else None
        )

class HanoiMDAP(MicroAgent):
    """MDAP implementation for Towers of Hanoi"""
    
    def __init__(self, config: MDAPConfig = None):
        if config is None:
            config = MDAPConfig(
                model="gpt-4o-mini",
                k_margin=3,
                max_candidates=10,
                temperature=0.1,  # Default temperature set to 0.1
                max_response_length=1000  # Keep default at 1000
            )
        super().__init__(config)
        self.harness = MDAPHarness(self.config)
    
    def create_initial_state(self, num_disks: int) -> HanoiState:
        """Create initial Hanoi state with all disks on peg A"""
        pegs = {
            'A': list(range(num_disks, 0, -1)),  # Largest to smallest
            'B': [],
            'C': []
        }
        return HanoiState(pegs=pegs, num_disks=num_disks, move_history=[])
    
    def generate_step_prompt(self, state: HanoiState) -> str:
        """Generate prompt for next step using the new MDAP prompt format"""
        # Get the previous move from the state
        previous_move = "None"  # Default for first move
        
        # Check move_history first as it's the primary record of moves
        if state.move_history and len(state.move_history) > 0:
            last_move = state.move_history[-1]
            previous_move = f"[{last_move['disk_id']}, {last_move['from_peg']}, {last_move['to_peg']}]"
        # Fallback to previous_move for backward compatibility
        elif state.previous_move is not None:
            previous_move = f"[{state.previous_move[0]}, {state.previous_move[1]}, {state.previous_move[2]}]"
        
        # Convert pegs from dict to list format for the paper's format
        # Map A->0, B->1, C->2
        peg_list = []
        for peg_key in ['A', 'B', 'C']:
            peg_list.append(state.pegs[peg_key])
        
        # Use the exact user template from the paper
        prompt = USER_TEMPLATE.format(
            previous_move=previous_move,
            current_state=json.dumps(peg_list)
        )
        
        return prompt
    
    def is_valid_move(self, state: HanoiState, from_peg: str, to_peg: str) -> bool:
        """Check if a move is valid according to Hanoi rules"""
        if from_peg not in state.pegs or to_peg not in state.pegs:
            return False
        
        if not state.pegs[from_peg]:  # No disk to move
            return False
        
        if from_peg == to_peg:  # Can't move to same peg
            return False
        
        # Check if move follows size rule
        moving_disk = state.pegs[from_peg][-1]
        if state.pegs[to_peg]:  # Destination not empty
            top_disk = state.pegs[to_peg][-1]
            if moving_disk > top_disk:  # Can't place larger on smaller
                return False
        
        return True
    
    def update_state(self, current_state: HanoiState, step_result: dict) -> HanoiState:
        """Update Hanoi state based on move and validate against prediction"""
        move = step_result['move']
        predicted_state_dict = step_result['predicted_state']
        
        # Create a new state to modify
        new_state = current_state.copy()
        
        # Convert from paper's format (list of lists) to internal format (dict)
        if isinstance(predicted_state_dict.get('pegs'), list):
            # Convert list format to dict format
            pegs_list = predicted_state_dict['pegs']
            new_state.pegs = {
                'A': pegs_list[0],
                'B': pegs_list[1],
                'C': pegs_list[2]
            }
        else:
            # Use dict format directly
            new_state.pegs = predicted_state_dict['pegs']
        
        # Handle move_count - increment from current state if not provided
        if 'move_count' in predicted_state_dict:
            new_state.move_count = predicted_state_dict['move_count']
        else:
            # If move_count is not in predicted_state, increment from current state
            new_state.move_count = current_state.move_count + 1
        
        # Initialize move_history if not present
        if new_state.move_history is None:
            new_state.move_history = []
        
        # Add move to history in paper's format
        new_state.move_history.append({
            'disk_id': move[0],
            'from_peg': move[1], 
            'to_peg': move[2]
        })
        
        return new_state
    
    def is_solved(self, state: HanoiState) -> bool:
        """Check if Hanoi is solved (all disks on peg C)"""
        return (
            len(state.pegs['C']) == state.num_disks and
            len(state.pegs['A']) == 0 and
            len(state.pegs['B']) == 0
        )
    
    def get_optimal_move(self, state: HanoiState) -> List[int]:
        """
        Get the optimal move for the current state based on the optimal strategy.
        Returns the move as [disk_id, from_peg, to_peg].
        """
        # Check if state is already solved
        if self.is_solved(state):
            return None
            
        # Map peg names to indices for easier calculation
        peg_indices = {'A': 0, 'B': 1, 'C': 2}
        
        # Determine the move based on the optimal strategy
        # Rule 1: If the previous move was NOT disk 1, move disk 1.
        # Rule 2: If the previous move WAS disk 1, make the only other legal move.
        
        previous_move_was_disk_1 = False
        # Check previous_move first (for calibration states)
        if state.previous_move is not None and len(state.previous_move) > 0:
            if state.previous_move[0] == 1:
                previous_move_was_disk_1 = True
        # Fallback to move_history for backward compatibility
        elif state.move_history and len(state.move_history) > 0:
            last_move = state.move_history[-1]
            if last_move['disk_id'] == 1:
                previous_move_was_disk_1 = True

        if not previous_move_was_disk_1:
            # --- Rule 1: Move disk 1 ---
            # Find the current peg of disk 1
            disk_1_peg_name = None
            for peg_name, peg in state.pegs.items():
                if peg and peg[-1] == 1:
                    disk_1_peg_name = peg_name
                    break
            
            if disk_1_peg_name is None:
                return None # Should not happen in a valid state

            from_idx = peg_indices[disk_1_peg_name]
            
            # Determine direction based on the number of disks
            if state.num_disks % 2 == 0:  # Even number of disks: counter-clockwise (A->C->B->A)
                to_idx = (from_idx - 1) % 3
            else:  # Odd number of disks: clockwise (A->B->C->A)
                to_idx = (from_idx + 1) % 3
                
            return [1, from_idx, to_idx]

        else:
            # --- Rule 2: Make the only other legal move ---
            # When previous move was disk 1, make the only legal move that doesn't involve disk 1
            # Find all possible moves that don't involve disk 1
            valid_moves = []
            
            for from_peg_name, peg in state.pegs.items():
                if not peg:
                    continue
                # Skip moves involving disk 1
                if peg[-1] == 1:
                    continue
                
                moving_disk = peg[-1]
                
                # Check all possible destination pegs
                for to_peg_name, dest_peg in state.pegs.items():
                    if from_peg_name == to_peg_name:
                        continue
                    
                    # Check if move is valid (can't place larger on smaller)
                    if dest_peg and moving_disk > dest_peg[-1]:
                        continue
                    
                    # This is a valid move
                    valid_moves.append((moving_disk, from_peg_name, to_peg_name))
            
            if not valid_moves:
                # No valid moves found (shouldn't happen in a valid unsolved state)
                return None
            
            # There should be exactly one valid move when following optimal strategy
            # If there are multiple, take the first one
            moving_disk, from_peg_name, to_peg_name = valid_moves[0]
            
            from_idx = peg_indices[from_peg_name]
            to_idx = peg_indices[to_peg_name]
            return [moving_disk, from_idx, to_idx]
    
    def step_generator(self, state: HanoiState) -> Tuple[Tuple[str, str], Callable]:
        """Generate prompt and parser for current step"""
        user_prompt = self.generate_step_prompt(state)
        # Format system prompt with configured token limit
        # Note: We had to add token limit requirement to SYSTEM_PROMPT for GLM 4.6
        system_prompt = SYSTEM_PROMPT.format(token_limit=self.config.max_response_length)
        parser = self.harness.red_flag_parser.parse_move_state_flag
        logger.info(f"Generated step prompt for state with move_count={state.move_count}")
        logger.info(f"System prompt length: {len(system_prompt)} characters")
        logger.info(f"User prompt length: {len(user_prompt)} characters")
        logger.info(f"Parser function: {parser}")
        return (system_prompt, user_prompt), parser
    
    async def solve_hanoi(self, num_disks: int) -> List[HanoiState]:
        """Solve Towers of Hanoi using MDAP"""
        logger.info(f"Starting to solve {num_disks}-disk Towers of Hanoi problem")
        logger.info(f"Using model: {self.config.model}, k_margin: {self.config.k_margin}")
        trace = await self.harness.execute_agent_mdap(self, num_disks)
        
        # Verify the final state is actually solved
        if trace and not self.is_solved(trace[-1]):
            logger.error(f"Solver completed but final state is not solved: {trace[-1].to_dict()}")
            raise RuntimeError("Hanoi solver failed to reach goal state")
        
        logger.info(f"Successfully solved {num_disks}-disk Hanoi in {trace[-1].move_count} moves")
        logger.info(f"Solution trace contains {len(trace)} states")
        return trace

# Utility function for testing
def print_solution(trace: List[HanoiState]):
    """Print the solution trace"""
    logger.info(f"Towers of Hanoi Solution ({len(trace)-1} moves):")
    for i, state in enumerate(trace):
        logger.info(f"\nStep {i}:")
        for peg, disks in state.pegs.items():
            logger.info(f"  {peg}: {disks}")
    logger.info(f"\nSolved in {trace[-1].move_count} moves!")
