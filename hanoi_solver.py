"""
Towers of Hanoi solver using MDAP harness
Demonstrates MAKER framework on a classic recursive problem
"""

import json
from typing import List, Tuple, Callable, Any
from dataclasses import dataclass
import copy
from micro_agent import MicroAgent
from mdap_harness import MDAPHarness, MDAPConfig, RedFlagParser

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
        prompt = f"""You are a Hanoi move generator. Respond with ONLY the move and next_state. No explanations, no reasoning, no other text.

Solve Towers of Hanoi. Move all disks to peg C.

1. Move top disk only.
2. Never place larger on smaller.

Current State:
Peg A: {state.pegs['A']}
Peg B: {state.pegs['B']}
Peg C: {state.pegs['C']}

Your task:
Choose the next move and predict the resulting state.

move = {{"from_peg": "X", "to_peg": "Y"}}
next_state = {{"pegs": {{"A": [...], "B": [...], "C": [...]}}, "num_disks": {state.num_disks}, "move_count": {state.move_count + 1}}}

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
        
        new_state = current_state.copy()
        from_peg = move['from_peg']
        to_peg = move['to_peg']
        
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
        return await self.harness.execute_agent_mdap(self, num_disks)

# Utility function for testing
def print_solution(trace: List[HanoiState]):
    """Print the solution trace"""
    print(f"Towers of Hanoi Solution ({len(trace)-1} moves):")
    for i, state in enumerate(trace):
        print(f"\nStep {i}:")
        for peg, disks in state.pegs.items():
            print(f"  {peg}: {disks}")
    print(f"\nSolved in {trace[-1].move_count} moves!")
