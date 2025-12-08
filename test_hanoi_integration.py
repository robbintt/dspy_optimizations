"""
Integration tests for Hanoi MDAP solver
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
from hanoi_solver import HanoiMDAP, HanoiState, MDAPConfig

class TestHanoiState:
    """Test HanoiState class"""
    
    def test_create_initial_state(self):
        """Test creating initial Hanoi state"""
        state = HanoiState(
            pegs={'A': [3, 2, 1], 'B': [], 'C': []},
            num_disks=3,
            move_count=0
        )
        
        assert state.pegs['A'] == [3, 2, 1]
        assert state.pegs['B'] == []
        assert state.pegs['C'] == []
        assert state.num_disks == 3
        assert state.move_count == 0
    
    def test_copy_state(self):
        """Test copying Hanoi state"""
        original = HanoiState(
            pegs={'A': [3, 2, 1], 'B': [], 'C': []},
            num_disks=3,
            move_count=5
        )
        
        copied = original.copy()
        
        # Verify copy is equal but not same object
        assert copied.pegs == original.pegs
        assert copied.num_disks == original.num_disks
        assert copied.move_count == original.move_count
        assert copied is not original
        assert copied.pegs is not original.pegs
    
    def test_to_dict(self):
        """Test converting state to dictionary"""
        state = HanoiState(
            pegs={'A': [2, 1], 'B': [3], 'C': []},
            num_disks=3,
            move_count=2
        )
        
        result = state.to_dict()
        expected = {
            'pegs': {'A': [2, 1], 'B': [3], 'C': []},
            'num_disks': 3,
            'move_count': 2
        }
        
        assert result == expected

class TestHanoiMDAP:
    """Test HanoiMDAP class"""
    
    @pytest.fixture
    def solver(self):
        """Create Hanoi solver for testing"""
        config = MDAPConfig(
            model="test-model",
            k_margin=2,
            max_candidates=5,
            temperature=0.1,
            max_retries=2
        )
        return HanoiMDAP(config)
    
    def test_create_initial_state(self, solver):
        """Test creating initial Hanoi state"""
        state = solver.create_initial_state(3)
        
        assert state.pegs['A'] == [3, 2, 1]
        assert state.pegs['B'] == []
        assert state.pegs['C'] == []
        assert state.num_disks == 3
        assert state.move_count == 0
    
    def test_create_initial_state_different_sizes(self, solver):
        """Test creating initial state with different disk counts"""
        for n in range(1, 6):
            state = solver.create_initial_state(n)
            assert state.pegs['A'] == list(range(n, 0, -1))
            assert state.pegs['B'] == []
            assert state.pegs['C'] == []
            assert state.num_disks == n
            assert state.move_count == 0
    
    def test_generate_step_prompt(self, solver):
        """Test generating step prompt"""
        state = HanoiState(
            pegs={'A': [3, 2], 'B': [1], 'C': []},
            num_disks=3,
            move_count=1
        )
        
        prompt = solver.generate_step_prompt(state)
        
        assert "3 disks" in prompt
        assert "Peg A: [3, 2]" in prompt
        assert "Peg B: [1]" in prompt
        assert "Peg C: []" in prompt
        assert '"from_peg": "A"' in prompt
        assert '"to_peg": "B"' in prompt
    
    def test_is_valid_move(self, solver):
        """Test move validation"""
        state = HanoiState(
            pegs={'A': [3, 2, 1], 'B': [], 'C': []},
            num_disks=3
        )
        
        # Valid moves
        assert solver.is_valid_move(state, 'A', 'B') == True
        assert solver.is_valid_move(state, 'A', 'C') == True
        
        # Invalid moves
        assert solver.is_valid_move(state, 'B', 'A') == False  # Empty source
        assert solver.is_valid_move(state, 'A', 'A') == False  # Same peg
        assert solver.is_valid_move(state, 'D', 'A') == False  # Invalid peg
        assert solver.is_valid_move(state, 'A', 'D') == False  # Invalid peg
    
    def test_is_valid_move_size_rule(self, solver):
        """Test move validation with size rule"""
        state = HanoiState(
            pegs={'A': [3], 'B': [2], 'C': [1]},
            num_disks=3
        )
        
        # Valid: move 1 onto empty
        assert solver.is_valid_move(state, 'C', 'B') == False  # 1 onto 2 is invalid
        assert solver.is_valid_move(state, 'C', 'A') == False  # 1 onto 3 is invalid
        assert solver.is_valid_move(state, 'B', 'C') == True   # 2 onto 1 is invalid
        assert solver.is_valid_move(state, 'B', 'A') == True   # 2 onto 3 is invalid
        assert solver.is_valid_move(state, 'A', 'B') == True   # 3 onto 2 is invalid
        assert solver.is_valid_move(state, 'A', 'C') == True   # 3 onto 1 is invalid
    
    def test_update_state_valid_move(self, solver):
        """Test updating state with valid move"""
        initial_state = HanoiState(
            pegs={'A': [3, 2, 1], 'B': [], 'C': []},
            num_disks=3,
            move_count=0
        )
        
        move = {"from_peg": "A", "to_peg": "B"}
        new_state = solver.update_state(initial_state, move)
        
        assert new_state.pegs['A'] == [3, 2]
        assert new_state.pegs['B'] == [1]
        assert new_state.pegs['C'] == []
        assert new_state.move_count == 1
        assert new_state.num_disks == 3
    
    def test_update_state_invalid_move(self, solver):
        """Test updating state with invalid move raises error"""
        initial_state = HanoiState(
            pegs={'A': [3, 2, 1], 'B': [], 'C': []},
            num_disks=3
        )
        
        # Invalid move: same peg
        move = {"from_peg": "A", "to_peg": "A"}
        
        with pytest.raises(ValueError, match="Invalid move"):
            solver.update_state(initial_state, move)
    
    def test_is_solved_true(self, solver):
        """Test solved state detection"""
        solved_state = HanoiState(
            pegs={'A': [], 'B': [], 'C': [3, 2, 1]},
            num_disks=3
        )
        
        assert solver.is_solved(solved_state) == True
    
    def test_is_solved_false(self, solver):
        """Test unsolved state detection"""
        unsolved_states = [
            HanoiState(pegs={'A': [1], 'B': [], 'C': [3, 2]}, num_disks=3),
            HanoiState(pegs={'A': [], 'B': [1], 'C': [3, 2]}, num_disks=3),
            HanoiState(pegs={'A': [3, 2, 1], 'B': [], 'C': []}, num_disks=3),
        ]
        
        for state in unsolved_states:
            assert solver.is_solved(state) == False
    
    def test_step_generator(self, solver):
        """Test step generator returns prompt and parser"""
        state = HanoiState(
            pegs={'A': [2, 1], 'B': [], 'C': [3]},
            num_disks=3
        )
        
        prompt, parser = solver.step_generator(state)
        
        assert isinstance(prompt, str)
        assert "2 disks" in prompt
        assert callable(parser)
    
    @pytest.mark.asyncio
    async def test_solve_hanoi_mock_success(self, solver):
        """Test solving Hanoi with mocked LLM responses"""
        # Mock the voting to return optimal moves for 2-disk Hanoi
        optimal_moves = [
            {"from_peg": "A", "to_peg": "B"},
            {"from_peg": "A", "to_peg": "C"},
            {"from_peg": "B", "to_peg": "C"}
        ]
        
        with patch.object(solver, 'first_to_ahead_by_k') as mock_voting:
            mock_voting.side_effect = optimal_moves
            
            trace = await solver.solve_hanoi(2)
            
            assert len(trace) == 4  # Initial + 3 moves
            assert solver.is_solved(trace[-1]) == True
            assert trace[-1].move_count == 3
            
            # Verify final state
            final_state = trace[-1]
            assert final_state.pegs['A'] == []
            assert final_state.pegs['B'] == []
            assert final_state.pegs['C'] == [2, 1]
    
    @pytest.mark.asyncio
    async def test_solve_hanoi_with_invalid_move(self, solver):
        """Test solving Hanoi when LLM returns invalid move"""
        # Mock responses: first invalid, then valid
        with patch.object(solver, 'first_to_ahead_by_k') as mock_voting:
            mock_voting.side_effect = [
                {"from_peg": "A", "to_peg": "A"},  # Invalid: same peg
                {"from_peg": "A", "to_peg": "B"},  # Valid
                {"from_peg": "A", "to_peg": "C"},
                {"from_peg": "B", "to_peg": "C"}
            ]
            
            with pytest.raises(ValueError, match="Invalid move"):
                await solver.solve_hanoi(2)

class TestHanoiIntegration:
    """End-to-end integration tests for Hanoi solver"""
    
    @pytest.mark.asyncio
    async def test_3_disk_hanoi_solution_structure(self):
        """Test that 3-disk Hanoi solution has correct structure"""
        config = MDAPConfig(k_margin=2, max_candidates=3)
        solver = HanoiMDAP(config)
        
        # Mock optimal solution
        optimal_moves = [
            {"from_peg": "A", "to_peg": "C"},
            {"from_peg": "A", "to_peg": "B"},
            {"from_peg": "C", "to_peg": "B"},
            {"from_peg": "A", "to_peg": "C"},
            {"from_peg": "B", "to_peg": "A"},
            {"from_peg": "B", "to_peg": "C"},
            {"from_peg": "A", "to_peg": "C"}
        ]
        
        with patch.object(solver, 'first_to_ahead_by_k') as mock_voting:
            mock_voting.side_effect = optimal_moves
            
            trace = await solver.solve_hanoi(3)
            
            # Verify solution structure
            assert len(trace) == 8  # Initial + 7 moves
            assert trace[0].move_count == 0
            assert trace[-1].move_count == 7
            
            # Verify all states are valid
            for i, state in enumerate(trace):
                assert isinstance(state, HanoiState)
                assert state.num_disks == 3
                assert state.move_count == i
                
                # Verify peg invariants
                total_disks = sum(len(pegs) for pegs in state.pegs.values())
                assert total_disks == 3
                
                # Verify disk ordering on each peg
                for peg, disks in state.pegs.items():
                    for i in range(len(disks) - 1):
                        assert disks[i] > disks[i + 1]  # Larger below smaller
            
            # Verify solution is solved
            assert solver.is_solved(trace[-1])
    
    @pytest.mark.asyncio
    async def test_execution_trace_consistency(self):
        """Test that execution trace maintains consistency"""
        config = MDAPConfig(k_margin=1, max_candidates=2)
        solver = HanoiMDAP(config)
        
        # Simple 1-disk solution
        with patch.object(solver, 'first_to_ahead_by_k') as mock_voting:
            mock_voting.return_value = {"from_peg": "A", "to_peg": "C"}
            
            trace = await solver.solve_hanoi(1)
            
            assert len(trace) == 2
            
            # Initial state
            assert trace[0].pegs == {'A': [1], 'B': [], 'C': []}
            assert trace[0].move_count == 0
            
            # Final state
            assert trace[1].pegs == {'A': [], 'B': [], 'C': [1]}
            assert trace[1].move_count == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
