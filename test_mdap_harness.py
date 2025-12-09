"""
Unit tests for MDAP Harness core functionality
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock
from collections import Counter
from mdap_harness import MDAPHarness, MDAPConfig, RedFlagParser
from micro_agent import MicroAgent

class TestMDAPConfig:
    """Test MDAPConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = MDAPConfig()
        assert config.model == "cerebras/zai-glm-4.6"
        assert config.k_margin == 6  # Updated default from paper's findings
        assert config.max_candidates == 10
        assert config.temperature == 0.1
        assert config.max_retries == 3
        assert config.cost_threshold is None
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = MDAPConfig(
            model="gpt-4o-mini",
            k_margin=5,
            max_candidates=15,
            temperature=0.2,
            max_retries=5,
            cost_threshold=10.0
        )
        assert config.model == "gpt-4o-mini"
        assert config.k_margin == 5
        assert config.max_candidates == 15
        assert config.temperature == 0.2
        assert config.max_retries == 5
        assert config.cost_threshold == 10.0

class TestRedFlagParser:
    """Test RedFlagParser class"""
    
    def test_valid_move_response(self):
        """Test parsing a valid move response in paper's format"""
        response = """move = [1, 0, 1]
next_state = {"pegs": [[], [1], []], "num_disks": 1, "move_count": 1}"""
        result = RedFlagParser.parse_move_state_flag(response)
        
        assert result is not None
        assert result['move'] == [1, 0, 1]
        assert result['predicted_state']['pegs'] == [[], [1], []]
    
    def test_valid_move_response_dict(self):
        """Test parsing a valid move response as dict (legacy format)"""
        response = {"from_peg": "A", "to_peg": "C"}
        result = RedFlagParser.parse_move_state_flag(response)
        
        assert result is not None
        assert result['move']['from_peg'] == 'A'
        assert result['move']['to_peg'] == 'C'
    
    def test_invalid_json(self):
        """Test parsing invalid JSON"""
        response = '{"from_peg": "A", "to_peg": "B"'  # Missing closing brace
        result = RedFlagParser.parse_move_state_flag(response)
        assert result is None
    
    def test_non_dict_response(self):
        """Test parsing non-dict response"""
        response = '"not a dict"'
        result = RedFlagParser.parse_move_state_flag(response)
        assert result is None
    
    def test_missing_fields(self):
        """Test parsing response with missing fields"""
        response = 'move = [1, 0]'  # Missing to_peg
        result = RedFlagParser.parse_move_state_flag(response)
        assert result is None
    
    def test_none_fields(self):
        """Test parsing response with None fields"""
        response = 'move = [1, null, 1]'
        result = RedFlagParser.parse_move_state_flag(response)
        assert result is None
    
    def test_invalid_peg_values(self):
        """Test parsing response with invalid peg values"""
        response = 'move = [1, 3, 1]'  # Peg 3 is not valid
        result = RedFlagParser.parse_move_state_flag(response)
        assert result is None
    
    def test_same_peg_move(self):
        """Test parsing response moving to same peg"""
        response = 'move = [1, 0, 0]'
        result = RedFlagParser.parse_move_state_flag(response)
        assert result is None
    
    def test_too_long_response(self):
        """Test parsing overly long response"""
        response = 'move = [1, 0, 1]\n' + 'x' * 1000
        result = RedFlagParser.parse_move_state_flag(response)
        assert result is None

class TestMDAPHarness:
    """Test MDAPHarness class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return MDAPConfig(
            model="gpt-4o-mini",
            k_margin=2,
            max_candidates=5,
            temperature=0.1,
            max_retries=2
        )
    
    @pytest.fixture
    def harness(self, config):
        """Create test harness"""
        return MDAPHarness(config)
    
    @pytest.mark.asyncio
    async def test_first_to_ahead_by_k_winner_found(self, harness):
        """Test first-to-ahead-by-K when winner is found"""
        # Mock responses
        mock_responses = [
            """move = [1, 0, 1]
next_state = {"pegs": [[], [1], []], "num_disks": 1, "move_count": 1}""",  # Valid
            """move = [1, 0, 1]
next_state = {"pegs": [[], [1], []], "num_disks": 1, "move_count": 1}""",  # Same valid response
            """move = [1, 0, 2]
next_state = {"pegs": [[], [], [1]], "num_disks": 1, "move_count": 1}""",  # Different valid response
        ]
        
        with patch('mdap_harness.acompletion') as mock_acompletion:
            # Setup mock to return responses in sequence
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = mock_responses[0]
            mock_acompletion.return_value = mock_response
            
            # First call returns first response
            result = await harness.first_to_ahead_by_k(
                "test prompt", 
                RedFlagParser.parse_move_state_flag
            )
            
            assert result['move'] == [1, 0, 1]
    
    @pytest.mark.asyncio
    async def test_first_to_ahead_by_k_red_flagged(self, harness):
        """Test first-to-ahead-by-K with red-flagged responses"""
        # Mock responses - first few are invalid, then valid
        mock_responses = [
            'invalid json',  # Invalid JSON
            """move = [1, 0, 0]
next_state = {"pegs": [[1], [], []], "num_disks": 1, "move_count": 1}""",  # Same peg move
            """move = [1, 0, 1]
next_state = {"pegs": [[], [1], []], "num_disks": 1, "move_count": 1}""",  # Valid
            """move = [1, 0, 1]
next_state = {"pegs": [[], [1], []], "num_disks": 1, "move_count": 1}""",  # Same valid response
            """move = [1, 0, 1]
next_state = {"pegs": [[], [1], []], "num_disks": 1, "move_count": 1}""",  # Same valid response
        ]
        
        with patch('mdap_harness.acompletion') as mock_acompletion:
            # Setup mock to return responses in sequence
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            
            call_count = 0
            def side_effect(*args, **kwargs):
                nonlocal call_count
                if call_count < len(mock_responses):
                    mock_response.choices[0].message.content = mock_responses[call_count]
                else:
                    mock_response.choices[0].message.content = '{"from_peg": "A", "to_peg": "B"}'
                call_count += 1
                return mock_response
            
            mock_acompletion.side_effect = side_effect
            
            # Add timeout to prevent hanging
            result = await asyncio.wait_for(
                harness.first_to_ahead_by_k(
                    "test prompt", 
                    RedFlagParser.parse_move_state_flag
                ),
                timeout=10.0
            )
            
            assert result['move'] == [1, 0, 1]
    
    @pytest.mark.asyncio
    async def test_first_to_ahead_by_k_no_valid_candidates(self, harness):
        """Test first-to-ahead-by-K when no valid candidates found"""
        with patch('mdap_harness.acompletion') as mock_acompletion:
            # Always return invalid response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = 'invalid json'
            mock_acompletion.return_value = mock_response
            
            # Reduce max_candidates to make the test fail faster
            original_max_candidates = harness.config.max_candidates
            harness.config.max_candidates = 3
            
            try:
                with pytest.raises(Exception, match="No valid candidates found"):
                    await harness.first_to_ahead_by_k(
                        "test prompt", 
                        RedFlagParser.parse_move_state_flag
                    )
            finally:
                # Restore original config
                harness.config.max_candidates = original_max_candidates
    
    @pytest.mark.asyncio
    async def test_execute_step_success(self, harness):
        """Test successful step execution"""
        with patch.object(harness, 'first_to_ahead_by_k') as mock_voting:
            mock_voting.return_value = {"move": [1, 0, 1], "predicted_state": {"pegs": [[], [1], []], "num_disks": 1, "move_count": 1}}
            
            result = await harness.execute_step(
                "test prompt",
                RedFlagParser.parse_move_state_flag
            )
            
            assert result['move'] == [1, 0, 1]
            mock_voting.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_step_retry_success(self, harness):
        """Test step execution with retry on failure"""
        with patch.object(harness, 'first_to_ahead_by_k') as mock_voting:
            # Fail first time, succeed second time
            mock_voting.side_effect = [
                Exception("First failure"),
                {"move": [1, 0, 1], "predicted_state": {"pegs": [[], [1], []], "num_disks": 1, "move_count": 1}}
            ]
            
            result = await harness.execute_step(
                "test prompt",
                RedFlagParser.parse_move_state_flag
            )
            
            assert result['move'] == [1, 0, 1]
            assert mock_voting.call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_step_max_retries_exceeded(self, harness):
        """Test step execution when max retries exceeded"""
        with patch.object(harness, 'first_to_ahead_by_k') as mock_voting:
            mock_voting.side_effect = Exception("Always fails")
            
            with pytest.raises(Exception, match="Step execution failed after 2 attempts"):
                await harness.execute_step(
                    "test prompt",
                    RedFlagParser.parse_move_state_flag
                )
            
            assert mock_voting.call_count == 2  # max_retries = 2
    
    def test_update_state_not_implemented(self, harness):
        """Test that update_state raises NotImplementedError"""
        with pytest.raises(NotImplementedError, match="Subclasses must implement update_state"):
            harness.update_state({}, {})
    
    @pytest.mark.asyncio
    async def test_execute_agent_mdap(self, harness):
        """Test executing MDAP with a micro agent"""
        class TestAgent(MicroAgent):
            def create_initial_state(self, max_steps):
                return {"step": 0, "max_steps": max_steps}
            
            def generate_step_prompt(self, state):
                return f"Current step: {state['step']}"
            
            def update_state(self, current_state, step_result):
                return {"step": current_state["step"] + 1, "max_steps": current_state["max_steps"]}
            
            def is_solved(self, state):
                return state["step"] >= state["max_steps"]
        
        agent = TestAgent()
        
        # Mock the LLM calls to avoid actual API calls
        with patch('mdap_harness.acompletion') as mock_acompletion:
            # Setup mock to return a simple response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "step completed"
            mock_acompletion.return_value = mock_response
            
            trace = await harness.execute_agent_mdap(agent, 3)
            
            assert len(trace) == 4  # Initial + 3 steps
            assert trace[0]["step"] == 0
            assert trace[-1]["step"] == 3

class TestMDAPCalibration:
    """Tests for the calibration functions"""

    @pytest.fixture
    def harness(self):
        """Create test harness for calibration tests"""
        return MDAPHarness(MDAPConfig(
            model="test-model",
            k_margin=2,
            max_candidates=5,
            temperature=0.1,
            max_retries=2
        ))

    @pytest.mark.asyncio
    async def test_estimate_per_step_success_rate(self, harness):
        """Test the per-step success rate estimation"""
        class MockAgent(MicroAgent):
            def create_initial_state(self, *args, **kwargs):
                return {'step': 0, 'max_steps': 100}
            
            def generate_step_prompt(self, state):
                return f"Step {state['step']}"
            
            def update_state(self, current_state, step_result):
                # The step_result is the parsed response from RedFlagParser
                # It should be a dict with 'move' and 'predicted_state' keys
                # Don't increment step here - the estimation function counts successful LLM calls
                if step_result and 'move' in step_result:
                    return current_state
                else:
                    raise ValueError("Invalid step result")
            
            def is_solved(self, state):
                return state['step'] >= state['max_steps']
            
            def step_generator(self, state):
                return self.generate_step_prompt(state), RedFlagParser.parse_move_state_flag

        # Mock the first_to_ahead_by_k to simulate a 70% success rate
        # The estimation function stops at the first failure, so we need
        # to ensure we get exactly 7 successes before any failure
        # Pattern: 7 successes, then failures
        success_pattern = [True] * 7 + [False, False, False]
        call_count = 0
        async def mock_first_to_ahead_by_k(prompt, parser):
            nonlocal call_count
            if call_count < len(success_pattern):
                result = success_pattern[call_count]
                call_count += 1
                if result:
                    # Return a valid response that the parser will accept
                    response = {"from_peg": "A", "to_peg": "B"}
                    return parser(response)
                else:
                    raise Exception("Simulated LLM failure")
            else:
                # Default to success for any additional calls
                response = {"from_peg": "A", "to_peg": "B"}
                return parser(response)
        
        harness.first_to_ahead_by_k = mock_first_to_ahead_by_k
        agent = MockAgent()
        
        p_estimate = await harness.estimate_per_step_success_rate(agent, num_disks=3, sample_steps=10)
        
        # The estimation stops at the first failure, so with 7 successes then failure:
        # it will attempt 8 steps (7 successes, 1 failure) and get 7/8 = 0.875
        # This is the actual behavior of the estimation function
        assert p_estimate == 0.875  # 7 successes out of 8 attempts before first failure

    def test_calculate_k_min(self, harness):
        """Test the k_min calculation with known values"""
        # Test with a high success rate
        k_min = harness.calculate_k_min(p=0.9, num_disks=3, target_reliability=0.95)
        # For 3 disks, s=7. With p=0.9, k should be small
        assert k_min >= 1
        
        # Test with a lower success rate
        k_min_low = harness.calculate_k_min(p=0.6, num_disks=4, target_reliability=0.95)
        # For 4 disks, s=15. With p=0.6, k should be larger
        assert k_min_low > k_min
        
        # Test edge case with p <= 0.5
        k_min_edge = harness.calculate_k_min(p=0.5, num_disks=3, target_reliability=0.95)
        assert k_min_edge == 20  # Should return the high default value
        
        # Test with p=1 (perfect model)
        k_min_perfect = harness.calculate_k_min(p=1.0, num_disks=3, target_reliability=0.95)
        # Should handle the edge case and return a reasonable default
        assert k_min_perfect >= 1

    @pytest.mark.asyncio
    async def test_estimate_per_step_success_rate_zero_success(self, harness):
        """Test estimation when model fails all steps"""
        class MockAgent(MicroAgent):
            def create_initial_state(self, *args, **kwargs):
                return {'step': 0}
            
            def generate_step_prompt(self, state):
                return "Always fail"
            
            def update_state(self, current_state, step_result):
                return current_state
            
            def is_solved(self, state):
                return False
            
            def step_generator(self, state):
                return self.generate_step_prompt(state), lambda x: {'move': 'ok'}

        # Mock to always fail
        async def mock_first_to_ahead_by_k(prompt, parser):
            raise Exception("Always fails")
        
        harness.first_to_ahead_by_k = mock_first_to_ahead_by_k
        agent = MockAgent()
        
        p_estimate = await harness.estimate_per_step_success_rate(agent, num_disks=3, sample_steps=10)
        
        assert p_estimate == 0.0


class TestMDAPIntegration:
    """Integration tests for MDAP framework"""
    
    @pytest.mark.asyncio
    async def test_simple_mdap_execution(self):
        """Test simple MDAP execution with mock LLM"""
        config = MDAPConfig(k_margin=2, max_candidates=3)
        harness = MDAPHarness(config)
        
        # Track state and steps
        states = [{"count": 0}, {"count": 1}, {"count": 2}]
        current_step = 0
        
        def step_generator(state):
            nonlocal current_step
            if current_step < len(states) - 1:
                prompt = f"Current count: {state['count']}. Increment by 1."
                parser = lambda x: {"increment": 1} if x == "increment" else None
                current_step += 1
                return prompt, parser
            return None, None
        
        def termination_check(state):
            return state["count"] >= 2
        
        def update_state(current_state, step_result):
            return {"count": current_state["count"] + step_result["increment"]}
        
        harness.update_state = update_state
        
        # Mock the voting to return predictable results
        with patch.object(harness, 'first_to_ahead_by_k') as mock_voting:
            mock_voting.return_value = {"increment": 1}
            
            trace = await harness.execute_mdap(
                initial_state={"count": 0},
                step_generator=step_generator,
                termination_check=termination_check
            )
            
            assert len(trace) == 3
            assert trace[0]["count"] == 0
            assert trace[1]["count"] == 1
            assert trace[2]["count"] == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
