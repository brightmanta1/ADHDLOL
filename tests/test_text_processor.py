import unittest
from models.text_processor import TextProcessor

class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = TextProcessor()
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

    def test_topic_extraction(self):
        """Test topic and subtopic extraction"""
        result = self.processor.process_text(self.test_content)
        
        # Verify topics structure
        self.assertIn("Understanding ADHD", result.topics)
        self.assertEqual(len(result.topics["Understanding ADHD"]), 3)
        self.assertIn("Executive Function Challenges", result.topics["Understanding ADHD"])

    def test_term_highlighting(self):
        """Test term highlighting with colors"""
        result = self.processor.process_text(self.test_content)
        
        # Verify highlighted terms
        self.assertIn("Time Management", result.highlighted_terms)
        self.assertIn("Hyperfocus", result.highlighted_terms)
        
        # Verify color assignments
        self.assertEqual(result.highlighted_terms["Time Management"], self.processor.term_colors["key_concept"])

    def test_tag_extraction(self):
        """Test content tagging"""
        result = self.processor.process_text(self.test_content)
        
        # Verify extracted tags
        self.assertIn("Understanding ADHD", result.tags)
        self.assertGreater(len(result.tags), 0)

    def test_complexity_analysis(self):
        """Test complexity scoring"""
        result = self.processor.process_text(self.test_content)
        
        # Verify complexity score
        self.assertGreater(result.complexity_score, 0)
        self.assertLess(result.complexity_score, 1)

if __name__ == '__main__':
    unittest.main()
