"""
Towers of Hanoi solver using MDAP harness
Demonstrates MAKER framework on a classic recursive problem
"""

import json
import logging
from typing import List, Tuple, Callable, Any
from dataclasses import dataclass
import copy
from micro_agent import MicroAgent
from mdap_harness import MDAPHarness, MDAPConfig, RedFlagParser

logger = logging.getLogger(__name__)

@dataclass
class HanoiState:
    """State representation for Towers of Hanoi"""
    pegs: dict  # {'A': [largest...smallest], 'B': [...], 'C': [...]}
    num_disks: int
    move_count: int = 0
    
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
            move_count=self.move_count
        )

class HanoiMDAP(MicroAgent):
    """MDAP implementation for Towers of Hanoi"""
    
    def __init__(self, config: MDAPConfig = None):
        if config is None:
            config = MDAPConfig(
                model="gpt-4o-mini",
                k_margin=3,
                max_candidates=10,
                temperature=0.1
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
        return HanoiState(pegs=pegs, num_disks=num_disks)
    
    def generate_step_prompt(self, state: HanoiState) -> str:
        """Generate prompt for the next move"""
        # Determine which peg has disks and which are empty
        disks_on_A = len(state.pegs['A'])
        disks_on_B = len(state.pegs['B'])
        disks_on_C = len(state.pegs['C'])
        
        # Get top disk on each peg if it exists
        top_A = state.pegs['A'][-1] if state.pegs['A'] else "empty"
        top_B = state.pegs['B'][-1] if state.pegs['B'] else "empty"
        top_C = state.pegs['C'][-1] if state.pegs['C'] else "empty"
        
        # Log the current state for debugging
        logger.info(f"CURRENT STATE BEFORE LLM CALL: A={state.pegs['A']}, B={state.pegs['B']}, C={state.pegs['C']}, move_count={state.move_count}")
        
        prompt = f"""You are a Hanoi move generator. You have {self.config.thinking_budget} tokens for thinking if needed.

Respond with ONLY the move and next_state. Use thinking tokens if the model supports them, but keep the final response concise.

Solve Towers of Hanoi. Move all disks to peg C.

CRITICAL DISK SIZE RULE:
- Disk numbers represent SIZE: 3 is BIGGER than 2, 2 is BIGGER than 1
- BIGGER disks can NEVER go on SMALLER disks
- VALID: [2, 1], [3, 1], [3, 2] (smaller on larger)
- INVALID: [1, 2], [1, 3], [2, 3] (larger on smaller)
- The peg list shows [bottom...top], so RIGHTMOST number is the TOP disk

FINAL ORDER ON PEG C:
- For 2 disks: [2, 1] - disk 2 at BOTTOM, disk 1 on TOP
- For 3 disks: [3, 2, 1] - disk 3 at BOTTOM, disk 2 in middle, disk 1 on TOP

GOAL: All disks on peg C in descending order [largest...smallest]
Goal State: Peg A: [], Peg B: [], Peg C: {list(range(state.num_disks, 0, -1))}

STRATEGY: 
- ALWAYS look at the CURRENT state and choose the BEST move for THIS situation
- If the largest disk can move to C, do it
- Otherwise, move smaller disks out of the way
- Think step-by-step: "What move helps me get closer to all disks on C?"

RULES:
1. Only move the TOP disk (rightmost number in the list)
2. NEVER place a LARGER disk on a SMALLER disk
3. Check destination peg's top disk before moving
4. Cannot move to the same peg

CURRENT STATE ANALYSIS:
- Peg A: {state.pegs['A']} (top: {top_A})
- Peg B: {state.pegs['B']} (top: {top_B})
- Peg C: {state.pegs['C']} (top: {top_C})

Current State (move {state.move_count}):
Peg A: {state.pegs['A']}
Peg B: {state.pegs['B']}
Peg C: {state.pegs['C']}

PROGRESS: {disks_on_C}/{state.num_disks} disks on goal peg

VALID MOVE CHECK:
Before choosing, verify: if destination has disks, is moving disk < destination top disk?

Your task:
Analyze the CURRENT state and choose the SMARTEST valid move.
Don't follow a pattern - think about what helps most right now.

move = {{"from_peg": "X", "to_peg": "Y"}}
next_state = {{"pegs": {{"A": {state.pegs['A']}, "B": {state.pegs['B']}, "C": {state.pegs['C']}}}, "num_disks": {state.num_disks}, "move_count": {state.move_count + 1}}}

REMEMBER: Respond with ONLY the two lines above. Nothing else."""
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
        from_peg = move['from_peg']
        to_peg = move['to_peg']
        
        # Check if the move is valid and apply it
        if self.is_valid_move(current_state, from_peg, to_peg):
            # Make the move
            disk = new_state.pegs[from_peg].pop()
            new_state.pegs[to_peg].append(disk)
            new_state.move_count += 1
        else:
            raise ValueError(f"Invalid move: {from_peg} -> {to_peg}")
        
        # Validate the LLM's prediction of the next state
        actual_state_dict = new_state.to_dict()
        if actual_state_dict != predicted_state_dict:
            raise ValueError(f"LLM's predicted state does not match actual state. Predicted: {predicted_state_dict}, Actual: {actual_state_dict}")
        
        return new_state
    
    def is_solved(self, state: HanoiState) -> bool:
        """Check if Hanoi is solved (all disks on peg C)"""
        return (
            len(state.pegs['C']) == state.num_disks and
            len(state.pegs['A']) == 0 and
            len(state.pegs['B']) == 0
        )
    
    def step_generator(self, state: HanoiState) -> Tuple[str, Callable]:
        """Generate prompt and parser for current step"""
        prompt = self.generate_step_prompt(state)
        parser = RedFlagParser.parse_move_state_flag
        return prompt, parser
    
    async def solve_hanoi(self, num_disks: int) -> List[HanoiState]:
        """Solve Towers of Hanoi using MDAP"""
        trace = await self.harness.execute_agent_mdap(self, num_disks)
        
        # Verify the final state is actually solved
        if trace and not self.is_solved(trace[-1]):
            logger.error(f"Solver completed but final state is not solved: {trace[-1].to_dict()}")
            raise RuntimeError("Hanoi solver failed to reach goal state")
        
        logger.info(f"Successfully solved {num_disks}-disk Hanoi in {trace[-1].move_count} moves")
        return trace

# Utility function for testing
def print_solution(trace: List[HanoiState]):
    """Print the solution trace"""
    print(f"Towers of Hanoi Solution ({len(trace)-1} moves):")
    for i, state in enumerate(trace):
        print(f"\nStep {i}:")
        for peg, disks in state.pegs.items():
            print(f"  {peg}: {disks}")
    print(f"\nSolved in {trace[-1].move_count} moves!")
