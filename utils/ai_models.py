import os
import re
from typing import Dict, List, Any
import random
import json

class SimpleAIModel:
    def __init__(self):
        self.templates = {
            "quiz": [
                "What is the main concept of {}?",
                "Which of the following best describes {}?",
                "How would you apply {} in practice?"
            ]
        }

    def simplify_content(self, text: str, complexity: str = "medium") -> Dict[str, Any]:
        """
        Simplify text content using basic rules.
        """
        try:
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            # Apply complexity-based formatting
            if complexity == "simple":
                # Break into very short chunks
                formatted = []
                for sentence in sentences:
                    words = sentence.split()
                    if len(words) > 8:
                        # Break long sentences
                        chunks = [' '.join(words[i:i+8]) for i in range(0, len(words), 8)]
                        formatted.extend(chunks)
                    else:
                        formatted.append(sentence)

                content = "• " + "\n• ".join(formatted)

            elif complexity == "medium":
                # Keep original sentences with bullet points
                content = "• " + "\n• ".join(sentences)

            else:  # advanced
                # Keep original format
                content = text

            return {
                "content": content,
                "stats": {
                    "original_length": len(text),
                    "num_sentences": len(sentences),
                    "complexity_level": complexity
                }
            }
        except Exception as e:
            return {"error": str(e), "original_text": text}

    def analyze_learning_style(self, user_interactions: Dict) -> Dict[str, Any]:
        """
        Analyze user's learning patterns without ML.
        """
        try:
            # Simple rule-based analysis
            style = "visual"  # default

            if user_interactions.get("preferred_display_mode") == "Audio":
                style = "auditory"
            elif user_interactions.get("completed_modules", 0) > 5:
                style = "self-paced"

            return {
                "preferred_style": style,
                "recommendations": [
                    "Break content into smaller chunks",
                    "Use visual aids when possible",
                    "Take frequent breaks"
                ]
            }
        except Exception as e:
            return {"error": str(e)}

    def generate_quiz(self, content: str, difficulty: str = "medium") -> Dict[str, Any]:
        """
        Generate quiz using templates.
        """
        try:
            # Extract key phrases (simple approach)
            words = content.split()
            key_phrases = [' '.join(words[i:i+3]) for i in range(0, len(words), 3)]

            questions = []
            for i in range(min(3, len(key_phrases))):
                phrase = key_phrases[i]
                template = random.choice(self.templates["quiz"])

                # Generate options
                correct_answer = phrase
                wrong_answers = random.sample(
                    [p for p in key_phrases if p != phrase],
                    min(3, len(key_phrases) - 1)
                )

                options = wrong_answers + [correct_answer]
                random.shuffle(options)

                questions.append({
                    "question": template.format(phrase),
                    "options": options,
                    "correct_answer": correct_answer
                })

            return {"questions": questions}
        except Exception as e:
            return {"error": str(e)}

    def adapt_content(self, text: str, learning_style: str) -> Dict[str, Any]:
        """
        Adapt content based on learning style.
        """
        try:
            if learning_style == "visual":
                # Add emoji indicators
                content = text.replace("Key points:", "📌 Key points:")
                content = content.replace("Example:", "💡 Example:")
            elif learning_style == "auditory":
                # Format for text-to-speech
                content = text.replace("-", " dash ")
                content = text.replace("/", " or ")
            else:
                content = text

            return {
                "adapted_content": content,
                "style_applied": learning_style
            }
        except Exception as e:
            return {"error": str(e)}

# Create a singleton instance
ai_model = SimpleAIModel()

# Export functions that match the original API
def simplify_content(text: str, complexity: str = "medium") -> Dict[str, Any]:
    return ai_model.simplify_content(text, complexity)

def analyze_learning_style(user_interactions: Dict) -> Dict[str, Any]:
    return ai_model.analyze_learning_style(user_interactions)

def create_personalized_quiz(content: str, difficulty: str = "medium") -> Dict[str, Any]:
    return ai_model.generate_quiz(content, difficulty)

def generate_adaptive_content(text: str, learning_style: str) -> Dict[str, Any]:
    return ai_model.adapt_content(text, learning_style)