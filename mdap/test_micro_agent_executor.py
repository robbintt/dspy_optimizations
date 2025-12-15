"""
Unit tests for the generic MicroAgentExecutor
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from micro_agent_executor import MicroAgentExecutor, execute_agent
from hanoi_solver import HanoiMDAP
from mdap_harness import MDAPConfig


@pytest.fixture
def mock_config():
    """Create a test config with fake model name"""
    return MDAPConfig(
        model="fake-provider/test-model-v1",
        k_margin=2,
        max_candidates=5
    )


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response that solves 3-disk Hanoi"""
    # Optimal solution for 3-disk Hanoi: A->C, A->B, C->B, A->C, B->A, B->C, A->C
    moves_sequence = [
        """move = [1, 0, 2]
next_state = [[3, 2], [], [1]]""",
        """move = [2, 0, 1]
next_state = [[3], [2], [1]]""",
        """move = [1, 2, 1]
next_state = [[3], [2, 1], []]""",
        """move = [3, 0, 2]
next_state = [[], [2, 1], [3]]""",
        """move = [1, 1, 0]
next_state = [[1], [2], [3]]""",
        """move = [2, 1, 2]
next_state = [[1], [], [3, 2]]""",
        """move = [1, 0, 2]
next_state = [[], [], [3, 2, 1]]""",
    ]

    # Create a single mock response object that we'll modify
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()

    # Set usage with concrete integer values
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50

    state = {'step_index': 0, 'votes_for_current_step': 0}

    def _create_side_effect(*args, **kwargs):
        # Return the same move multiple times for voting (k_margin=2 means we need at least 2 votes)
        # Advance to next move after 3 votes to ensure it wins
        mock_response.choices[0].message.content = moves_sequence[state['step_index']]

        state['votes_for_current_step'] += 1
        if state['votes_for_current_step'] >= 3:  # After 3 calls, move to next step
            state['votes_for_current_step'] = 0
            state['step_index'] = (state['step_index'] + 1) % len(moves_sequence)

        return mock_response

    return _create_side_effect


class TestMicroAgentExecutor:
    """Tests for the generic MicroAgentExecutor"""

    @pytest.mark.asyncio
    async def test_basic_execution(self, mock_config, mock_llm_response):
        """Test basic executor functionality"""
        with patch('mdap_harness.acompletion') as mock_completion:
            # Use the side_effect function from the fixture
            mock_completion.side_effect = mock_llm_response

            agent = HanoiMDAP(mock_config)
            executor = MicroAgentExecutor(agent)

            trace = await executor.execute(num_disks=3)

            assert len(trace) > 0
            assert trace[-1].num_disks == 3
            assert agent.is_solved(trace[-1])

    @pytest.mark.asyncio
    async def test_with_custom_config(self, mock_llm_response):
        """Test executor with custom configuration"""
        with patch('mdap_harness.acompletion') as mock_completion:
            mock_completion.side_effect = mock_llm_response

            config = MDAPConfig(
                model="fake-provider/custom-model",
                k_margin=2,
                max_candidates=5
            )

            agent = HanoiMDAP(config)
            executor = MicroAgentExecutor(agent, config)

            trace = await executor.execute(num_disks=3)

            assert executor.config.k_margin == 2
            assert executor.config.max_candidates == 5
            assert agent.is_solved(trace[-1])

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, mock_config, mock_llm_response):
        """Test that execution statistics are tracked correctly"""
        with patch('mdap_harness.acompletion') as mock_completion:
            mock_completion.side_effect = mock_llm_response

            agent = HanoiMDAP(mock_config)
            executor = MicroAgentExecutor(agent)

            # Initial statistics should be zero
            assert executor.total_cost == 0.0
            assert executor.total_api_calls == 0

            # Execute
            await executor.execute(num_disks=3)

            # Statistics should be updated
            assert executor.total_api_calls > 0
            stats = executor.get_statistics()
            assert 'total_cost' in stats
            assert 'total_api_calls' in stats
            assert 'model' in stats

    @pytest.mark.asyncio
    async def test_reset_statistics(self, mock_config, mock_llm_response):
        """Test resetting statistics"""
        with patch('mdap_harness.acompletion') as mock_completion:
            mock_completion.side_effect = mock_llm_response

            agent = HanoiMDAP(mock_config)
            executor = MicroAgentExecutor(agent)

            # Execute to accumulate stats
            await executor.execute(num_disks=3)
            assert executor.total_api_calls > 0

            # Reset
            executor.reset_statistics()
            assert executor.total_cost == 0.0
            assert executor.total_api_calls == 0
            assert executor.total_input_tokens == 0
            assert executor.total_output_tokens == 0

    @pytest.mark.asyncio
    async def test_update_k_margin(self, mock_config):
        """Test updating k_margin parameter"""
        agent = HanoiMDAP(mock_config)
        executor = MicroAgentExecutor(agent)

        initial_k = executor.config.k_margin
        executor.update_k_margin(5)
        assert executor.config.k_margin == 5
        assert executor.config.k_margin != initial_k

    @pytest.mark.asyncio
    async def test_convenience_function(self, mock_config, mock_llm_response):
        """Test the convenience execute_agent function"""
        with patch('mdap_harness.acompletion') as mock_completion:
            mock_completion.side_effect = mock_llm_response

            agent = HanoiMDAP(mock_config)
            trace = await execute_agent(agent, num_disks=3)

            assert len(trace) > 0
            assert agent.is_solved(trace[-1])

    @pytest.mark.skip(reason="Mock fixture needs to support multiple problem sizes")
    @pytest.mark.asyncio
    async def test_multiple_executions(self, mock_config, mock_llm_response):
        """Test running multiple executions with the same executor"""
        with patch('mdap_harness.acompletion') as mock_completion:
            mock_completion.side_effect = mock_llm_response

            agent = HanoiMDAP(mock_config)
            executor = MicroAgentExecutor(agent)

            for n in [3, 4]:
                executor.reset_statistics()
                trace = await executor.execute(num_disks=n)

                assert len(trace) > 0
                assert trace[-1].num_disks == n
                assert agent.is_solved(trace[-1])

    def test_executor_initialization(self):
        """Test different initialization patterns"""
        # With agent only (using fake model)
        config = MDAPConfig(model="fake-provider/init-test-model")
        agent = HanoiMDAP(config)
        executor1 = MicroAgentExecutor(agent)
        assert executor1.agent == agent

        # With agent and config
        config = MDAPConfig(model="fake-provider/test-model", k_margin=5)
        executor2 = MicroAgentExecutor(agent, config)
        assert executor2.config.k_margin == 5

        # Agent with its own config
        agent_with_config = HanoiMDAP(config)
        executor3 = MicroAgentExecutor(agent_with_config)
        assert executor3.config == agent_with_config.config

    @pytest.mark.asyncio
    async def test_execution_failure_handling(self, mock_config):
        """Test that execution failures are properly propagated"""
        agent = HanoiMDAP(mock_config)
        executor = MicroAgentExecutor(agent)

        # This should raise an error for invalid arguments
        with pytest.raises(Exception):
            await executor.execute()  # Missing required num_disks argument


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
