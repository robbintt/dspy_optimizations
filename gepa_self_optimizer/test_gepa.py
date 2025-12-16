import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import torch

# Import the actual classes and functions to be tested
import dspy
from gepa_config import _create_lm, JUDGE_CONSTITUTION
from gepa_system import GlmSelfReflect, Generate, ShepherdCritic, Refine
from generate_data import generate_synthetic_data, TopicToQA, BugInjector
from optimize_gepa import semantic_similarity, refinement_gepa_metric


class TestGepaConfig(unittest.TestCase):
    """Test the configuration module."""

    @patch('gepa_config.yaml.safe_load')
    @patch('gepa_config.open', new_callable=MagicMock)
    @patch('gepa_config.Path')
    @patch('gepa_config.os.getenv', return_value='MOCKED_API_KEY')
    def test_create_lm_success(self, mock_getenv, mock_path, mock_open_file, mock_yaml):
        """
        Test that `_create_lm` uses dspy.LM and then calls dspy.configure.
        """
        mock_config_data = {
            'test_config': {
                'provider': 'test_provider',
                'name': 'test_model',
                'temperature': 0.5,
                'max_tokens': 100
            }
        }
        mock_yaml.return_value = mock_config_data
        
        # Mock the dspy.LM class and the dspy.configure function
        with patch('gepa_config.dspy.LM') as mock_dspy_lm, \
             patch('gepa_config.dspy.configure') as mock_dspy_configure:
            
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
            
            # Assert that dspy.configure was called with the created LM instance
            mock_dspy_configure.assert_called_once_with(lm=mock_lm_instance)
            
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
        self.system.generator = Mock(spec=dspy.ChainOfThought)
        self.system.critic = Mock(spec=dspy.ChainOfThought)
        self.system.refiner = Mock(spec=dspy.Predict)

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
        mock_topic_to_qa_pred = Mock(question="q1", correct_answer="a1")
        mock_bug_injector_pred = Mock(bad_draft="bad_a1", gold_critique="error_desc1")
        
        with patch('generate_data.dspy.ChainOfThought') as mock_chain_of_thought:
            mock_chain_instance = mock_chain_of_thought.return_value
            mock_chain_instance.side_effect = [mock_topic_to_qa_pred, mock_bug_injector_pred]

            dataset = generate_synthetic_data(num_examples=1)
            self.assertEqual(len(dataset), 1)
            example = dataset[0]
            self.assertEqual(example.question, "q1")
            self.assertEqual(example.draft_answer, "bad_a1")
            self.assertEqual(example.gold_critique, "error_desc1")
            self.assertEqual(example.correct_answer, "a1")
            self.assertEqual(example.inputs(), {"question": "q1", "draft_answer": "bad_a1"})

    def test_generate_synthetic_data_handles_exceptions(self):
        """Test that data generation handles exceptions gracefully and continues."""
        with patch('generate_data.dspy.ChainOfThought', side_effect=Exception("Test error")):
            result = generate_synthetic_data(num_examples=1)
            self.assertEqual(len(result), 0)


class TestOptimizeGepa(unittest.TestCase):
    """Test the optimization module."""

    @patch('optimize_gepa.SentenceTransformer')
    @patch('optimize_gepa.util.cos_sim')
    def test_semantic_similarity(self, mock_cos_sim, mock_transformer):
        """Test semantic similarity calculation with mocked dependencies."""
        mock_embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        mock_model_instance = Mock()
        mock_model_instance.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model_instance
        mock_cos_sim.return_value = torch.tensor([[1.0]])
        
        similarity = semantic_similarity("text1", "text2")
        self.assertEqual(similarity, 1.0)
        mock_model_instance.encode.assert_called_once_with(["text1", "text2"], convert_to_tensor=True)
        mock_cos_sim.assert_called_once()

    @patch('optimize_gepa.semantic_similarity')
    @patch('optimize_gepa.dspy.evaluate.answer_with_feedback')
    def test_refinement_gepa_metric(self, mock_feedback_fn, mock_semantic_similarity):
        """Test the GEPA metric function."""
        gold = Mock()
        gold.correct_answer = "reference_answer"
        pred = Mock()
        pred.answer = "predicted_answer"
        
        mock_semantic_similarity.return_value = 0.85
        mock_feedback_obj = Mock()
        mock_feedback_fn.return_value = mock_feedback_obj

        result = refinement_gepa_metric(gold, pred)

        self.assertEqual(result, mock_feedback_obj)
        mock_semantic_similarity.assert_called_once_with("predicted_answer", "reference_answer")
        mock_feedback_fn.assert_called_once_with(0.85, "Similarity score is 0.85. The reference answer is 'reference_answer'.")


if __name__ == '__main__':
    unittest.main()
