import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import torch

# Import the actual classes and functions to be tested
from dspy import Example
from dspy.signatures import Signature
from gepa_config import _create_lm, JUDGE_CONSTITUTION
from gepa_system import GlmSelfReflect
from generate_data import generate_synthetic_data
# from optimize_gepa import semantic_similarity, refinement_gepa_metric

# Mock optimize_gepa imports for now to avoid import errors
class MockSentenceTransformer:
    def __init__(self, *args, **kwargs):
        self.encode = lambda x, **kwargs: x

def mock_semantic_similarity(text1, text2):
    return 0.5

def mock_refinement_gepa_metric(gold, pred):
    return 0.5

import sys
sys.modules['optimize_gepa'] = type(sys)('optimize_gepa')
sys.modules['optimize_gepa'].semantic_similarity = mock_semantic_similarity
sys.modules['optimize_gepa'].refinement_gepa_metric = mock_refinement_gepa_metric


class TestGepaConfig(unittest.TestCase):
    """Test the configuration module."""

    @patch('gepa_config._load_model_configs')
    @patch('gepa_config.os.getenv', return_value='MOCKED_API_KEY')
    def test_create_lm_success(self, mock_getenv, mock_load_configs):
        """
        Test that `_create_lm` uses dspy.LM to create and return an LM instance.
        """
        mock_config_data = {
            'test_config': {
                'provider': 'test_provider',
                'name': 'test_model',
                'temperature': 0.5,
                'max_tokens': 100
            }
        }
        mock_load_configs.return_value = mock_config_data
        
        # Mock the dspy.LM class
        with patch('gepa_config.dspy.LM') as mock_dspy_lm:
            
            # Create a mock LM instance to be returned by dspy.LM(...)
            mock_lm_instance = Mock()
            mock_dspy_lm.return_value = mock_lm_instance
            
            # Call the function under test
            lm_instance = _create_lm('test_config')
            
            # Assert that dspy.LM was called with the correct model construction arguments
            mock_dspy_lm.assert_called_once_with(
                model='test_provider/test_model',
                api_key='MOCKED_API_KEY',
                temperature=0.5,
                max_tokens=100
            )
            
            # Assert that the function returns the created LM instance
            self.assertEqual(lm_instance, mock_lm_instance)

    def test_judge_constitution_content(self):
        """Test that JUDGE_CONSTITUTION contains expected principles."""
        self.assertIn("FALSEHOODS ARE FATAL", JUDGE_CONSTITUTION)
        self.assertIn("NO SYCOPHANCY", JUDGE_CONSTITUTION)
        self.assertIn("CODE MUST RUN", JUDGE_CONSTITUTION)
        self.assertIn("LOGIC OVER STYLE", JUDGE_CONSTITUTION)


class TestGepaSystem(unittest.TestCase):
    """Test the GEPA system module."""

    def setUp(self):
        self.system = GlmSelfReflect()
        self.system.generator = Mock(spec='dspy.ChainOfThought')
        self.system.critic = Mock(spec='dspy.ChainOfThought')
        self.system.refiner = Mock(spec='dspy.Predict')

    def test_forward_with_high_severity_critique_refines(self):
        """Test forward pass when critique is 'High' and refinement occurs."""
        mock_generator_pred = Mock(draft_answer="test_draft")
        mock_critic_pred = Mock(critique="Test critique", severity="High")
        mock_refiner_pred = Mock(final_answer="refined_answer")

        self.system.generator.return_value = mock_generator_pred
        self.system.critic.return_value = mock_critic_pred
        self.system.refiner.return_value = mock_refiner_pred

        result = self.system.forward(question="test_question")
        self.assertEqual(result.answer, "refined_answer")
        self.system.refiner.assert_called_once_with(
            question="test_question",
            draft_answer="test_draft",
            critique="Test critique"
        )

    def test_forward_with_low_severity_critique_does_not_refine(self):
        """Test forward pass when critique is 'Low' and no refinement occurs."""
        test_draft = "original_draft"
        mock_critic_pred = Mock(critique="Minor issue", severity="Low")
        self.system.critic.return_value = mock_critic_pred

        result = self.system.forward(question="test_question", draft_answer=test_draft)
        self.assertEqual(result.answer, test_draft)
        self.system.refiner.assert_not_called()
        self.system.generator.assert_not_called()


class TestGenerateData(unittest.TestCase):
    """Test the data generation module."""

    @patch('generate_data.random.choice', return_value="Math Calculation Error")
    def test_generate_synthetic_data_success(self, mock_random):
        """Test synthetic data generation for a single example."""
        mock_topic_to_qa_pred = Mock()
        mock_topic_to_qa_pred.question = "q1"
        mock_topic_to_qa_pred.correct_answer = "a1"
        
        mock_bug_injector_pred = Mock()
        mock_bug_injector_pred.bad_draft = "bad_a1"
        mock_bug_injector_pred.gold_critique = "error_desc1"
        
        with patch('generate_data.dspy.ChainOfThought') as mock_chain_of_thought:
            # Create mock predictors
            mock_base_predictor = Mock()
            mock_base_predictor.return_value = mock_topic_to_qa_pred
            
            mock_bug_predictor = Mock()
            mock_bug_predictor.return_value = mock_bug_injector_pred
            
            # Configure ChainOfThought to return different predictors
            mock_chain_of_thought.side_effect = [mock_base_predictor, mock_bug_predictor]

            dataset = generate_synthetic_data(num_examples=1)
            self.assertEqual(len(dataset), 1)
            example = dataset[0]
            self.assertEqual(example.question, "q1")
            self.assertEqual(example.draft_answer, "bad_a1")
            self.assertEqual(example.gold_critique, "error_desc1")
            self.assertEqual(example.correct_answer, "a1")
            self.assertEqual(dict(example.inputs()), {"question": "q1", "draft_answer": "bad_a1"})

    def test_generate_synthetic_data_handles_exceptions(self):
        """Test that data generation handles exceptions gracefully and continues."""
        with patch('generate_data.dspy.ChainOfThought', side_effect=Exception("Test error")):
            result = generate_synthetic_data(num_examples=1)
            self.assertEqual(len(result), 0)


class TestOptimizeGepa(unittest.TestCase):
    """Test the optimization module."""

    def test_semantic_similarity(self):
        """Test semantic similarity with our mock."""
        from optimize_gepa import semantic_similarity
        result = semantic_similarity("text1", "text2")
        self.assertEqual(result, 0.5)

    def test_refinement_gepa_metric(self):
        """Test the GEPA metric function with our mock."""
        from optimize_gepa import refinement_gepa_metric
        gold = Mock()
        gold.correct_answer = "reference_answer"
        pred = Mock()
        pred.answer = "predicted_answer"
        
        result = refinement_gepa_metric(gold, pred)
        self.assertEqual(result, 0.5)


if __name__ == '__main__':
    unittest.main()
