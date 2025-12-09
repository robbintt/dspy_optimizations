"""
Towers of Hanoi solver using MDAP harness
Demonstrates MAKER framework on a classic recursive problem
"""

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
        # Create a visual representation of the state
        def visualize_pegs(pegs):
            max_height = max(len(p) for p in pegs.values())
            lines = []
            for level in range(max_height - 1, -1, -1):
                line_parts = []
                for peg_name in ['A', 'B', 'C']:
                    disks = pegs[peg_name]
                    if level < len(disks):
                        disk_size = disks[level]
                        line_parts.append(f"  {disk_size}  ")
                    else:
                        line_parts.append("  |  ")
                lines.append("".join(line_parts))
            lines.append("-----" * 3) # Base
            lines.append("  A    B    C  ")
            return "\n".join(lines)

        visual_state = visualize_pegs(state.pegs)
        
        prompt = f"""You are solving the Towers of Hanoi puzzle. Your goal is to move all disks to peg C.

**OVERALL STRATEGY:**
Follow this optimal strategy for {state.num_disks} disks:
1. Move the smallest disk (1) to the auxiliary peg (B).
2. Move the remaining disks (2 to {state.num_disks}) to the target peg (C).
3. Move the smallest disk (1) from the auxiliary peg (B) to the target peg (C).

**CRITICAL RULES:**
- Disk numbers represent SIZE: {state.num_disks} is LARGEST, 1 is SMALLEST.
- You can ONLY move the TOP disk from a peg.
- You can NEVER place a larger disk on top of a smaller disk.
- The goal is to move the entire tower to peg C.

**Current State (Move {state.move_count}):**
{visual_state}

**Analysis:**
- Top disk on A: {state.pegs['A'][-1] if state.pegs['A'] else 'None'}
- Top disk on B: {state.pegs['B'][-1] if state.pegs['B'] else 'None'}
- Top disk on C: {state.pegs['C'][-1] if state.pegs['C'] else 'None'}

**Your Task:**
Based on the OVERALL STRATEGY and the current state, choose the SINGLE next move. Do not undo the previous move.

Respond with ONLY a JSON object. No other text.
{{"from_peg": "A", "to_peg": "B"}}
"""
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
        """Update Hanoi state based on move"""
        new_state = current_state.copy()
        from_peg = step_result['from_peg']
        to_peg = step_result['to_peg']
        
        if self.is_valid_move(current_state, from_peg, to_peg):
            # Make the move
            disk = new_state.pegs[from_peg].pop()
            new_state.pegs[to_peg].append(disk)
            new_state.move_count += 1
        else:
            raise ValueError(f"Invalid move: {from_peg} -> {to_peg}")
        
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
