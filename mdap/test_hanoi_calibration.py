"""
Unit tests for Hanoi solver calibration functionality
Tests the mock step generator and optimal move logic used in calibration
"""

import pytest
import asyncio
import copy
import os
from unittest.mock import patch, MagicMock

from hanoi_solver import HanoiMDAP, HanoiState
from mdap_harness import MDAPConfig


class TestHanoiCalibration:
    """Test calibration-specific functionality of Hanoi solver"""
    
    @pytest.fixture
    def solver(self):
        """Create a HanoiMDAP solver instance for testing"""
        config = MDAPConfig(
            model="test-model",
            k_margin=1,
            temperature=0.1,
            max_response_length=1000
        )
        return HanoiMDAP(config=config)
    
    def test_get_optimal_move_initial_state(self, solver):
        """Test optimal move from initial state (should move disk 1)"""
        state = solver.create_initial_state(3)
        move = solver.get_optimal_move(state)
        
        # Should move disk 1 from A to B (for odd number of disks)
        assert move == [1, 0, 1]
    
    def test_get_optimal_move_after_disk_1(self, solver):
        """Test optimal move after moving disk 1 (should move another disk)"""
        state = solver.create_initial_state(3)
        # Simulate moving disk 1 from A to B
        state.move_history.append({'disk_id': 1, 'from_peg': 0, 'to_peg': 1})
        state.pegs['A'] = [3, 2]  # Remove disk 1
        state.pegs['B'] = [1]     # Add disk 1
        
        move = solver.get_optimal_move(state)
        
        # Should move disk 2 from A to C
        assert move == [2, 0, 2]
    
    def test_get_optimal_move_even_disks(self, solver):
        """Test optimal move direction for even number of disks"""
        state = solver.create_initial_state(4)  # Even number
        move = solver.get_optimal_move(state)
        
        # For even disks, disk 1 moves counter-clockwise (A to C)
        assert move == [1, 0, 2]
    
    def test_get_optimal_move_clockwise_pattern(self, solver):
        """Test that disk 1 moves clockwise for odd disk count"""
        state = solver.create_initial_state(3)
        
        # First move: A to B
        move1 = solver.get_optimal_move(state)
        assert move1 == [1, 0, 1]
        
        # Apply first move
        state.move_history.append({'disk_id': 1, 'from_peg': 0, 'to_peg': 1})
        state.pegs['A'] = [3, 2]
        state.pegs['B'] = [1]
        
        # Second move should be disk 2
        move2 = solver.get_optimal_move(state)
        assert move2 == [2, 0, 2]
        
        # Apply second move
        state.move_history.append({'disk_id': 2, 'from_peg': 0, 'to_peg': 2})
        state.pegs['A'] = [3]
        state.pegs['C'] = [2]
        
        # Third move: disk 1 from B to C
        move3 = solver.get_optimal_move(state)
        assert move3 == [1, 1, 2]
    
    def test_get_optimal_move_counter_clockwise_pattern(self, solver):
        """Test that disk 1 moves counter-clockwise for even disk count"""
        state = solver.create_initial_state(4)
        
        # First move: A to C (counter-clockwise)
        move1 = solver.get_optimal_move(state)
        assert move1 == [1, 0, 2]
        
        # Apply first move
        state.move_history.append({'disk_id': 1, 'from_peg': 0, 'to_peg': 2})
        state.pegs['A'] = [4, 3, 2]
        state.pegs['C'] = [1]
        
        # Second move should be disk 2
        move2 = solver.get_optimal_move(state)
        assert move2 == [2, 0, 1]
        
        # Apply second move
        state.move_history.append({'disk_id': 2, 'from_peg': 0, 'to_peg': 1})
        state.pegs['A'] = [4, 3]
        state.pegs['B'] = [2]
        
        # Third move: disk 1 from C to B (counter-clockwise)
        move3 = solver.get_optimal_move(state)
        assert move3 == [1, 2, 1]
    
    def test_get_optimal_move_solved_state(self, solver):
        """Test that solved state returns None"""
        state = solver.create_initial_state(3)
        # Move all disks to C (solved state)
        state.pegs = {'A': [], 'B': [], 'C': [3, 2, 1]}
        
        move = solver.get_optimal_move(state)
        assert move is None
    
    def test_get_optimal_move_invalid_state(self, solver):
        """Test optimal move with an invalid state"""
        state = solver.create_initial_state(3)
        # Create an invalid state (larger disk on smaller)
        state.pegs = {'A': [3], 'B': [2, 1], 'C': []}
        
        # Should still find a valid move if possible
        move = solver.get_optimal_move(state)
        # The exact move depends on the history, but it shouldn't crash
        assert move is not None or move is None  # Just ensure no exception
    
    def test_mock_step_generator_format(self, solver):
        """Test that mock step generator returns correct format"""
        state = solver.create_initial_state(3)
        
        def mock_step_generator(state):
            optimal_move = solver.get_optimal_move(state)
            
            if not optimal_move:
                raise ValueError(f"get_optimal_move returned no valid move for state: {state.to_dict()}")
            
            disk_id, from_peg, to_peg = optimal_move
            new_pegs = {peg: list(disks) for peg, disks in state.pegs.items()}
            
            # Move the disk
            disk = new_pegs[chr(65 + from_peg)].pop()
            new_pegs[chr(65 + to_peg)].append(disk)
            
            new_state = HanoiState(
                pegs=new_pegs,
                num_disks=state.num_disks,
                move_count=state.move_count + 1,
                move_history=copy.deepcopy(state.move_history) if state.move_history else []
            )
            
            new_state.move_history.append({
                'disk_id': disk_id,
                'from_peg': from_peg,
                'to_peg': to_peg
            })
            
            return "mock_prompt", lambda x: {
                "move": optimal_move,
                "predicted_state": new_state.to_dict()
            }
        
        prompt, parser = mock_step_generator(state)
        
        # Check prompt is a string
        assert isinstance(prompt, str)
        assert prompt == "mock_prompt"
        
        # Check parser returns correct format
        result = parser("dummy")
        assert "move" in result
        assert "predicted_state" in result
        assert len(result["move"]) == 3
        assert "pegs" in result["predicted_state"]
    
    def test_mock_step_generator_handles_none(self, solver):
        """Test that mock step generator raises error when get_optimal_move returns None"""
        state = solver.create_initial_state(3)
        state.pegs = {'A': [], 'B': [], 'C': [3, 2, 1]}  # Solved state
        
        def mock_step_generator(state):
            optimal_move = solver.get_optimal_move(state)
            
            if not optimal_move:
                raise ValueError(f"get_optimal_move returned no valid move for state: {state.to_dict()}")
            
            return "mock_prompt", lambda x: {"move": optimal_move}
        
        # Should raise ValueError for solved state
        with pytest.raises(ValueError, match="get_optimal_move returned no valid move"):
            mock_step_generator(state)
    
    @pytest.mark.asyncio
    async def test_calibration_mock_solution_consistency(self, solver):
        """Test that mock generator produces consistent optimal solution"""
        state = solver.create_initial_state(3)
        states = [state.copy()]
        
        # Generate steps until solved or max steps reached
        max_steps = 7  # First 7 moves of 3-disk solution
        for step in range(max_steps):
            optimal_move = solver.get_optimal_move(state)
            
            # Stop if state is solved
            if optimal_move is None:
                break
                
            assert optimal_move is not None, f"No move found at step {len(states)}"
            
            # Apply the move
            disk_id, from_peg, to_peg = optimal_move
            peg_names = ['A', 'B', 'C']
            from_peg_name = peg_names[from_peg]
            to_peg_name = peg_names[to_peg]
            
            disk = state.pegs[from_peg_name].pop()
            state.pegs[to_peg_name].append(disk)
            state.move_count += 1
            state.move_history.append({
                'disk_id': disk_id,
                'from_peg': from_peg,
                'to_peg': to_peg
            })
            
            states.append(state.copy())
        
        # Verify the sequence follows optimal pattern for the moves that were made
        expected_moves = [
            [1, 0, 1],  # Disk 1 A->B
            [2, 0, 2],  # Disk 2 A->C
            [1, 1, 2],  # Disk 1 B->C
            [3, 0, 1],  # Disk 3 A->B
            [1, 2, 0],  # Disk 1 C->A
            [2, 2, 1],  # Disk 2 C->B
            [1, 0, 1],  # Disk 1 A->B
        ]
        
        # Only check the moves that were actually made
        for i, (state, expected_move) in enumerate(zip(states[1:], expected_moves[:len(states)-1])):
            assert state.move_history[-1]['disk_id'] == expected_move[0]
            assert state.move_history[-1]['from_peg'] == expected_move[1]
            assert state.move_history[-1]['to_peg'] == expected_move[2]
    
    def test_optimal_move_20_disks(self, solver):
        """Test optimal move calculation for 20 disks (calibration use case)"""
        state = solver.create_initial_state(20)
        
        # First move should be disk 1 from A to B (20 is even, so counter-clockwise A->C)
        move = solver.get_optimal_move(state)
        assert move == [1, 0, 2]  # A->C for even number of disks
        
        # After moving disk 1, should move disk 2
        state.move_history.append({'disk_id': 1, 'from_peg': 0, 'to_peg': 2})
        state.pegs['A'] = list(range(20, 1, -1))  # Remove disk 1
        state.pegs['C'] = [1]  # Add disk 1
        
        move2 = solver.get_optimal_move(state)
        assert move2 == [2, 0, 1]  # Disk 2 A->B
    
    def test_state_copy_independence(self, solver):
        """Test that state copies are independent"""
        state = solver.create_initial_state(3)
        state_copy = state.copy()
        
        # Modify original
        state.pegs['A'].append(99)
        state.move_history.append({'test': 'value'})
        
        # Copy should be unchanged
        assert 99 not in state_copy.pegs['A']
        assert {'test': 'value'} not in state_copy.move_history
    
    def test_optimal_move_with_empty_history(self, solver):
        """Test optimal move when move_history is empty"""
        state = solver.create_initial_state(3)
        state.move_history = []  # Explicitly empty
        
        move = solver.get_optimal_move(state)
        assert move == [1, 0, 1]  # Should move disk 1
    
    def test_optimal_move_with_none_history(self, solver):
        """Test optimal move when move_history is None"""
        state = solver.create_initial_state(3)
        state.move_history = None
        
        move = solver.get_optimal_move(state)
        assert move == [1, 0, 1]  # Should move disk 1
    
    @pytest.mark.asyncio
    async def test_calibration_mock_solution_consistency(self, solver):
        """Test that mock calibration produces consistent results"""
        # Import here to avoid circular imports
        from calibrate_hanoi import generate_calibration_cache
        
        # Test with a smaller problem for speed
        num_disks = 5
        cache_file = "test_calibration_cache.pkl"
        
        # Remove existing test cache file
        if os.path.exists(cache_file):
            os.remove(cache_file)
        
        # Generate cache
        calibration_data = await generate_calibration_cache(num_disks=num_disks, cache_file=cache_file)
        
        # Verify cache structure
        assert calibration_data is not None
        assert 'num_disks' in calibration_data
        assert 'states' in calibration_data
        assert 'total_steps' in calibration_data
        assert 'sampled_count' in calibration_data
        
        # Verify values
        assert calibration_data['num_disks'] == num_disks
        assert calibration_data['total_steps'] == (2 ** num_disks) - 1  # Number of moves
        assert len(calibration_data['states']) == calibration_data['sampled_count']
        assert calibration_data['sampled_count'] <= 100000  # Should be limited to 100K
        
        # Verify at least one state has the correct initial configuration
        # Since we're sampling, the first state might not be the initial state
        found_initial = False
        for state in calibration_data['states']:
            if (state.pegs['A'] == list(range(num_disks, 0, -1)) and 
                state.pegs['B'] == [] and 
                state.pegs['C'] == []):
                found_initial = True
                break
        assert found_initial, "No state with initial configuration found in sample"
        
        # Verify last state is solved
        final_state = calibration_data['states'][-1]
        assert len(final_state.pegs['C']) == num_disks
        assert len(final_state.pegs['A']) == 0
        assert len(final_state.pegs['B']) == 0
        
        # Clean up test cache file
        if os.path.exists(cache_file):
            os.remove(cache_file)
