import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ProcessedText:
    original: str
    simplified: str
    complexity_score: float
    key_concepts: List[str]

class TextProcessor:
    def __init__(self):
        self.complexity_thresholds = {
            "simple": 0.4,
            "medium": 0.7,
            "advanced": 1.0
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

    def simplify_text(self, text: str, target_complexity: str = "medium") -> ProcessedText:
        """Simplify text based on target complexity level."""
        current_complexity = self.calculate_complexity(text)
        target_score = self.complexity_thresholds[target_complexity]
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        simplified_sentences = []
        for sentence in sentences:
            if self.calculate_complexity(sentence) > target_score:
                # Break into smaller chunks
                words = sentence.split()
                chunks = [' '.join(words[i:i+8]) for i in range(0, len(words), 8)]
                simplified_sentences.extend(chunks)
            else:
                simplified_sentences.append(sentence)
        
        simplified_text = '. '.join(simplified_sentences)
        key_concepts = self.extract_key_concepts(text)
        
        return ProcessedText(
            original=text,
            simplified=simplified_text,
            complexity_score=current_complexity,
            key_concepts=key_concepts
        )
