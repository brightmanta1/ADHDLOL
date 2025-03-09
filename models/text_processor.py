import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ProcessedText:
    original: str
    simplified: str
    complexity_score: float
    key_concepts: List[str]
    topics: Dict[str, List[str]]  # Topic -> Subtopics
    highlighted_terms: Dict[str, str]  # Term -> Color
    tags: List[str]

class TextProcessor:
    def __init__(self):
        self.complexity_thresholds = {
            "simple": 0.4,
            "medium": 0.7,
            "advanced": 1.0
        }
        self.term_colors = {
            "definition": "#2196F3",  # Blue
            "example": "#4CAF50",     # Green
            "key_concept": "#FF9800", # Orange
            "technique": "#9C27B0"    # Purple
        }

    def calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score based on sentence length and word complexity."""
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences) if sentences else 0

        # Normalize scores between 0 and 1
        word_complexity = min(avg_word_length / 10, 1)
        sentence_complexity = min(avg_sentence_length / 20, 1)

        return (word_complexity + sentence_complexity) / 2

    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts using basic NLP techniques."""
        sentences = re.split(r'[.!?]+', text)
        concepts = []

        for sentence in sentences:
            # Look for phrases after common markers
            markers = ["is", "are", "means", "refers to", "defined as"]
            for marker in markers:
                if marker in sentence.lower():
                    concept = sentence.lower().split(marker)[-1].strip()
                    if len(concept.split()) <= 5:  # Keep concepts concise
                        concepts.append(concept)

        return list(set(concepts))  # Remove duplicates

    def identify_topics(self, text: str) -> Dict[str, List[str]]:
        """Break down content into topics and subtopics."""
        topics = {}
        current_topic = ""
        current_subtopics = []

        # Split into lines and process
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Enhanced topic pattern matching
            topic_patterns = [
                r'^(?:Chapter|Section|Part)\s+\d+:\s*(.+)$',  # Chapter 1: Topic
                r'^(?:\d+\.|[A-Z][a-z]+:|\#)\s*(.+)$',       # 1. Topic or Topic: or #Topic
                r'^([A-Z][a-z\s]+(?:\s+[A-Z][a-z\s]+)*):',   # Title Case Topic:
            ]

            is_topic = False
            for pattern in topic_patterns:
                topic_match = re.match(pattern, line)
                if topic_match:
                    # Save previous topic if exists
                    if current_topic and current_subtopics:
                        topics[current_topic] = current_subtopics

                    current_topic = topic_match.group(1).strip()
                    current_subtopics = []
                    is_topic = True
                    break

            if not is_topic:
                # Check for subtopics (bullets, numbers, etc.)
                subtopic_match = re.match(r'^(?:-|\*|\d+\.\d+\.?|\•)\s*(.+)$', line)
                if subtopic_match and current_topic:
                    subtopic = subtopic_match.group(1).strip()
                    if subtopic:
                        current_subtopics.append(subtopic)

        # Save last topic
        if current_topic and current_subtopics:
            topics[current_topic] = current_subtopics

        return topics

    def highlight_terms(self, text: str) -> Dict[str, str]:
        """Identify and color-code important terms."""
        highlighted_terms = {}

        # Find definitions
        definition_pattern = r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+(?:is|are|refers to|means)'
        for match in re.finditer(definition_pattern, text):
            term = match.group(1)
            highlighted_terms[term] = self.term_colors["definition"]

        # Find examples
        example_pattern = r'(?:For example|Such as|Like)\s+([A-Z][a-z]+(?:\s+[a-z]+)*)'
        for match in re.finditer(example_pattern, text):
            term = match.group(1)
            highlighted_terms[term] = self.term_colors["example"]

        # Find key concepts (capitalized phrases)
        concept_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        for match in re.finditer(concept_pattern, text):
            term = match.group(1)
            if term not in highlighted_terms:
                highlighted_terms[term] = self.term_colors["key_concept"]

        return highlighted_terms

    def extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from the content."""
        tags = set()

        # Look for hashtags
        hashtags = re.findall(r'#(\w+)', text)
        tags.update(hashtags)

        # Look for keywords in parentheses
        keywords = re.findall(r'\(([^)]+)\)', text)
        tags.update(kw.strip() for kw in keywords)

        # Add topic names as tags
        topics = self.identify_topics(text)
        tags.update(topics.keys())

        # Add key terms
        highlighted_terms = self.highlight_terms(text)
        tags.update(highlighted_terms.keys())

        return list(tags)

    def process_text(self, text: str, target_complexity: str = "medium") -> ProcessedText:
        """Process text with full hierarchy and highlighting."""
        current_complexity = self.calculate_complexity(text)
        topics = self.identify_topics(text)
        highlighted_terms = self.highlight_terms(text)
        tags = self.extract_tags(text)
        key_concepts = self.extract_key_concepts(text)

        # Simplify if needed
        if current_complexity > self.complexity_thresholds[target_complexity]:
            simplified = self._simplify(text, target_complexity)
        else:
            simplified = text

        return ProcessedText(
            original=text,
            simplified=simplified,
            complexity_score=current_complexity,
            key_concepts=key_concepts,
            topics=topics,
            highlighted_terms=highlighted_terms,
            tags=tags
        )

    def _simplify(self, text: str, target_complexity: str) -> str:
        """Simplify text based on target complexity level."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Apply complexity-based formatting
        if target_complexity == "simple":
            # Break into very short chunks
            formatted = []
            for sentence in sentences:
                words = sentence.split()
                if len(words) > 8:
                    chunks = [' '.join(words[i:i+8]) for i in range(0, len(words), 8)]
                    formatted.extend(chunks)
                else:
                    formatted.append(sentence)

            result = '• ' + '\n• '.join(formatted)

        elif target_complexity == "medium":
            # Keep original sentences with bullet points
            result = '• ' + '\n• '.join(sentences)

        else:  # advanced
            result = text

        return result