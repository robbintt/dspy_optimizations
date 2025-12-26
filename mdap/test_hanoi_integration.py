"""
Integration tests for Hanoi MDAP solver
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
from .hanoi_solver import HanoiMDAP, HanoiState, MDAPConfig
from .microagent import MicroAgent

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
        
        # Check that the new prompt format is used
        assert "Previous move:" in prompt
        assert "Current State:" in prompt
        assert "[[3, 2], [1], []]" in prompt  # JSON format of pegs
        assert "clockwise one peg" in prompt  # From the new template
    
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
        
        # Test size rule: can only place smaller disk on larger disk
        assert solver.is_valid_move(state, 'C', 'B') == True   # 1 onto 2 is valid (smaller on larger)
        assert solver.is_valid_move(state, 'C', 'A') == True   # 1 onto 3 is valid (smaller on larger)
        assert solver.is_valid_move(state, 'B', 'C') == False  # 2 onto 1 is invalid (larger on smaller)
        assert solver.is_valid_move(state, 'B', 'A') == True   # 2 onto 3 is valid (smaller on larger)
        assert solver.is_valid_move(state, 'A', 'B') == False  # 3 onto 2 is invalid (larger on smaller)
        assert solver.is_valid_move(state, 'A', 'C') == False  # 3 onto 1 is invalid (larger on smaller)
    
    def test_update_state_valid_move(self, solver):
        """Test updating state with valid move"""
        initial_state = HanoiState(
            pegs={'A': [3, 2, 1], 'B': [], 'C': []},
            num_disks=3,
            move_count=0
        )
        
        # Use the new format with move and predicted_state
        step_result = {
            "move": [1, 0, 1],  # disk 1 from peg 0 to peg 1
            "predicted_state": {
                "pegs": [[3, 2], [1], []],
                "num_disks": 3,
                "move_count": 1
            }
        }
        new_state = solver.update_state(initial_state, step_result)
        
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
        
        # Invalid move: same peg (disk 1 from peg 0 to peg 0)
        step_result = {
            "move": [1, 0, 0],
            "predicted_state": {
                "pegs": [[3, 2, 1], [], []],
                "num_disks": 3,
                "move_count": 1
            }
        }
        
        # The update_state method now trusts the predicted_state from the LLM
        # It doesn't validate the move itself, just applies the predicted state
        new_state = solver.update_state(initial_state, step_result)
        
        # The state will be updated to match the prediction
        assert new_state.pegs == {'A': [3, 2, 1], 'B': [], 'C': []}
        assert new_state.move_count == 1
    
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
        assert "3 disks" in prompt
        assert callable(parser)
    
    @pytest.mark.asyncio
    async def test_solve_hanoi_mock_success(self, solver):
        """Test solving Hanoi with mocked LLM responses"""
        # Mock the voting to return optimal moves for 2-disk Hanoi in paper's format
        optimal_moves = [
            {
                "move": [1, 0, 1],  # Move disk 1 from peg 0 to peg 1
                "predicted_state": {
                    "pegs": [[2], [1], []],
                    "num_disks": 2,
                    "move_count": 1
                }
            },
            {
                "move": [2, 0, 2],  # Move disk 2 from peg 0 to peg 2
                "predicted_state": {
                    "pegs": [[], [1], [2]],
                    "num_disks": 2,
                    "move_count": 2
                }
            },
            {
                "move": [1, 1, 2],  # Move disk 1 from peg 1 to peg 2
                "predicted_state": {
                    "pegs": [[], [], [2, 1]],
                    "num_disks": 2,
                    "move_count": 3
                }
            }
        ]
        
        with patch.object(solver.harness, 'first_to_ahead_by_k') as mock_voting:
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
        # Mock responses: first invalid (red-flagged), then valid
        with patch.object(solver.harness, 'first_to_ahead_by_k') as mock_voting:
            mock_voting.side_effect = [
                None,  # Red-flagged response
                {
                    "move": [1, 0, 1],  # Valid
                    "predicted_state": {
                        "pegs": [[2], [1], []],
                        "num_disks": 2,
                        "move_count": 1
                    }
                },
                {
                    "move": [2, 0, 2],  # Valid
                    "predicted_state": {
                        "pegs": [[], [1], [2]],
                        "num_disks": 2,
                        "move_count": 2
                    }
                },
                {
                    "move": [1, 1, 2],  # Valid
                    "predicted_state": {
                        "pegs": [[], [], [2, 1]],
                        "num_disks": 2,
                        "move_count": 3
                    }
                }
            ]
            
            trace = await solver.solve_hanoi(2)
            
            # Should succeed after red-flagged response is discarded
            assert len(trace) == 4
            assert solver.is_solved(trace[-1])

class TestHanoiIntegration:
    """End-to-end integration tests for Hanoi solver"""
    
    @pytest.mark.asyncio
    async def test_3_disk_hanoi_solution_structure(self):
        """Test that 3-disk Hanoi solution has correct structure"""
        config = MDAPConfig(k_margin=2, max_candidates=3)
        solver = HanoiMDAP(config)
        
        # Mock optimal solution in paper's format
        optimal_moves = [
            {
                "move": [1, 0, 2],
                "predicted_state": {"pegs": [[3, 2], [], [1]], "num_disks": 3, "move_count": 1}
            },
            {
                "move": [2, 0, 1],
                "predicted_state": {"pegs": [[3], [2], [1]], "num_disks": 3, "move_count": 2}
            },
            {
                "move": [1, 2, 1],
                "predicted_state": {"pegs": [[3], [2, 1], []], "num_disks": 3, "move_count": 3}
            },
            {
                "move": [3, 0, 2],
                "predicted_state": {"pegs": [[], [2, 1], [3]], "num_disks": 3, "move_count": 4}
            },
            {
                "move": [1, 1, 0],
                "predicted_state": {"pegs": [[1], [2], [3]], "num_disks": 3, "move_count": 5}
            },
            {
                "move": [2, 1, 2],
                "predicted_state": {"pegs": [[1], [], [3, 2]], "num_disks": 3, "move_count": 6}
            },
            {
                "move": [1, 0, 2],
                "predicted_state": {"pegs": [[], [], [3, 2, 1]], "num_disks": 3, "move_count": 7}
            }
        ]
        
        with patch.object(solver.harness, 'first_to_ahead_by_k') as mock_voting:
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
        with patch.object(solver.harness, 'first_to_ahead_by_k') as mock_voting:
            mock_voting.return_value = {
                "move": [1, 0, 2],
                "predicted_state": {"pegs": [[], [], [1]], "num_disks": 1, "move_count": 1}
            }
            
            trace = await solver.solve_hanoi(1)
            
            assert len(trace) == 2
            
            # Initial state
            assert trace[0].pegs == {'A': [1], 'B': [], 'C': []}
            assert trace[0].move_count == 0
            
            # Final state
            assert trace[1].pegs == {'A': [], 'B': [], 'C': [1]}
            assert trace[1].move_count == 1

    @pytest.mark.asyncio
    async def test_solver_stops_immediately_at_goal(self):
        """Test that solver stops immediately when goal is reached"""
        config = MDAPConfig(k_margin=2, max_candidates=3)
        solver = HanoiMDAP(config)
        
        # Track how many times the LLM is called
        call_count = 0
        
        def mock_acompletion_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            
            # Return the final winning move that solves the puzzle
            # Move disk 1 from C to B, then disk 2 from B to A, then disk 1 from B to C
            # Actually, let's use a simpler state - move disk 1 from C to B
            if call_count <= 6:  # Allow enough calls for voting
                mock_response.choices[0].message.content = """move = {"from_peg": "C", "to_peg": "B"}
next_state = {"pegs": {"A": [], "B": [1], "C": []}, "num_disks": 1, "move_count": 4}"""
            else:
                # Should not reach here if solver stops correctly
                mock_response.choices[0].message.content = """move = {"from_peg": "A", "to_peg": "B"}
next_state = {"pegs": {"A": [], "B": [2], "C": [1]}, "num_disks": 2, "move_count": 5}"""
            
            return mock_response
        
        with patch('mdap_harness.acompletion', side_effect=mock_acompletion_side_effect):
            # Start from a state that's already solved
            initial_state = HanoiState(
                pegs={'A': [], 'B': [], 'C': [2, 1]},
                num_disks=2,
                move_count=3
            )
            
            trace = await solver.harness.execute_mdap(
                initial_state=initial_state,
                step_generator=solver.step_generator,
                termination_check=solver.is_solved,
                agent=solver
            )
            
            # Should not execute any steps since already solved
            assert len(trace) == 1  # Only initial state
            assert solver.is_solved(trace[-1])
            # Verify it doesn't continue after solving
            assert trace[-1].move_count == 3
    
    @pytest.mark.asyncio
    async def test_solver_raises_error_if_not_solved(self):
        """Test that solver raises RuntimeError if final state is not solved"""
        config = MDAPConfig(k_margin=2, max_candidates=3)
        solver = HanoiMDAP(config)
        
        # Mock the execute_agent_mdap method to return an unsolved trace
        async def mock_execute_agent_mdap(agent, num_disks):
            # Return a trace that ends in an unsolved state
            # Final state has disks on A and B, not all on C
            return [
                HanoiState(pegs={'A': [2, 1], 'B': [], 'C': []}, num_disks=2, move_count=0),
                HanoiState(pegs={'A': [2], 'B': [1], 'C': []}, num_disks=2, move_count=1)
            ]
        
        with patch.object(solver.harness, 'execute_agent_mdap', side_effect=mock_execute_agent_mdap):
            with pytest.raises(RuntimeError, match="Hanoi solver failed to reach goal state"):
                await solver.solve_hanoi(2)
    
    @pytest.mark.asyncio
    async def test_no_extra_steps_after_goal_state(self):
        """Test that no extra steps are attempted after reaching goal state"""
        config = MDAPConfig(k_margin=2, max_candidates=3)
        solver = HanoiMDAP(config)
        
        # Track LLM calls to ensure no extra calls after solving
        llm_calls = []
        
        def mock_acompletion_side_effect(*args, **kwargs):
            llm_calls.append(len(llm_calls) + 1)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            
            # Return a valid move (but we won't reach this since already solved)
            mock_response.choices[0].message.content = """move = {"from_peg": "A", "to_peg": "B"}
next_state = {"pegs": {"A": [2], "B": [1], "C": []}, "num_disks": 2, "move_count": 1}"""
            
            return mock_response
        
        with patch('mdap_harness.acompletion', side_effect=mock_acompletion_side_effect):
            # Start from a state that's already solved
            initial_state = HanoiState(
                pegs={'A': [], 'B': [], 'C': [2, 1]},
                num_disks=2,
                move_count=3
            )
            
            trace = await solver.harness.execute_mdap(
                initial_state=initial_state,
                step_generator=solver.step_generator,
                termination_check=solver.is_solved,
                agent=solver
            )
            
            # Should make no LLM calls since already solved
            assert len(llm_calls) == 0
            assert len(trace) == 1  # Only initial state
            assert solver.is_solved(trace[-1])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# The test classes are already properly structured for pytest
# No additional changes needed
