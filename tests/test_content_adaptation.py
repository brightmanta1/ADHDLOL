import unittest
from unittest.mock import patch, MagicMock
from models.advanced_ai import AdvancedAIModel
from models.text_processor import TextProcessor
from datetime import datetime

class TestContentAdaptation(unittest.TestCase):
    def setUp(self):
        # Mock Pinecone
        self.pinecone_mock = MagicMock()
        self.index_mock = MagicMock()

        # Mock database session
        self.session_mock = MagicMock()
        self.db_session_mock = MagicMock()
        self.db_session_mock.return_value.__enter__.return_value = self.session_mock

        with patch('pinecone.Pinecone', return_value=self.pinecone_mock) as mock_pinecone, \
             patch('models.advanced_ai.init_db', return_value=self.db_session_mock):
            self.pinecone_mock.list_indexes = MagicMock(return_value=MagicMock(names=lambda: []))
            self.pinecone_mock.Index = MagicMock(return_value=self.index_mock)
            self.ai_model = AdvancedAIModel()

        self.test_content = """
Chapter 1: Understanding ADHD
- Executive Function Challenges
- Working Memory Issues
- Focus Management

For example, Time Management is a key skill that affects daily tasks.
Hyperfocus refers to intense concentration periods.

Key points:
1. ADHD symptoms vary by person
2. Important strategies must be personalized
3. Regular breaks are critical for maintaining focus
"""

        # Set up test user data
        self.ai_model.track_user_interaction(
            user_id="test_user",
            content_id="test_content",
            interaction_type="study",
            duration=300,
            meta_data={
                "completion_rate": 0.8,
                "content_type": "lesson"
            }
        )

    def test_deep_focus_adaptation(self):
        """Test content adaptation for deep focus learning style"""
        # Mock learning pattern
        pattern = MagicMock()
        pattern.preferred_style = "deep_focus"
        self.session_mock.query().filter_by().first.return_value = pattern

        adapted_content = self.ai_model.adapt_content(
            content=self.test_content,
            user_id="test_user",
            complexity="medium"
        )

        # Verify structure
        self.assertIn("content", adapted_content)
        self.assertIn("topics", adapted_content)
        self.assertIn("highlighted_terms", adapted_content)
        self.assertIn("tags", adapted_content)

        # Verify content organization
        self.assertIn("Understanding ADHD", str(adapted_content["topics"]))
        self.assertIn("📚", adapted_content.get("content", ""))

    def test_adaptive_formatting(self):
        """Test different complexity levels of content adaptation"""
        for complexity in ["simple", "medium", "advanced"]:
            adapted_content = self.ai_model.adapt_content(
                content=self.test_content,
                user_id="test_user",
                complexity=complexity
            )

            # Verify adaptation properties
            self.assertIn("complexity_score", adapted_content)
            self.assertIn("content", adapted_content)
            self.assertGreater(len(adapted_content["key_concepts"]), 0)

    def test_content_hierarchy(self):
        """Test hierarchical content organization"""
        adapted_content = self.ai_model.adapt_content(
            content=self.test_content,
            user_id="test_user",
            complexity="medium"
        )

        # Verify hierarchy elements
        self.assertIsInstance(adapted_content["topics"], dict)
        self.assertIsInstance(adapted_content["highlighted_terms"], dict)
        self.assertIsInstance(adapted_content["tags"], list)

if __name__ == '__main__':
    unittest.main()