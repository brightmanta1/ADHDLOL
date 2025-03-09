import os
import pinecone
import assemblyai as aai
from datetime import datetime
import numpy as np
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
import json
import logging

from .database import init_db, UserInteraction, LearningPattern, ContentVector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAIModel:
    def __init__(self):
        logger.info("Initializing AdvancedAIModel...")
        try:
            # Initialize Pinecone
            pinecone.init(
                api_key=os.getenv('PINECONE_API_KEY'),
                environment=os.getenv('PINECONE_ENV')
            )
            self.index_name = "adhd-learning-content"
            if self.index_name not in pinecone.list_indexes():
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                pinecone.create_index(self.index_name, dimension=1536)
            self.vector_index = pinecone.Index(self.index_name)

            # Initialize AssemblyAI
            aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')

            # Initialize database
            self.DBSession = init_db()
            logger.info("AdvancedAIModel initialization completed successfully")

        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    def track_user_interaction(
        self,
        user_id: str,
        content_id: str,
        interaction_type: str,
        duration: float,
        metadata: Dict = None
    ) -> None:
        """Track user interactions with content."""
        try:
            logger.info(f"Tracking interaction for user {user_id}")
            with self.DBSession() as session:
                interaction = UserInteraction(
                    user_id=user_id,
                    content_id=content_id,
                    interaction_type=interaction_type,
                    duration=duration,
                    metadata=metadata or {}
                )
                session.add(interaction)
                session.commit()
                logger.info("Interaction tracked successfully")
        except Exception as e:
            logger.error(f"Error tracking interaction: {str(e)}")
            raise

    def analyze_learning_pattern(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's learning patterns using tracked interactions."""
        try:
            logger.info(f"Analyzing learning pattern for user {user_id}")
            with self.DBSession() as session:
                interactions = session.query(UserInteraction).filter_by(user_id=user_id).all()

                if not interactions:
                    logger.warning(f"No interaction data found for user {user_id}")
                    return {"error": "No interaction data found"}

                focus_durations = [i.duration for i in interactions if i.interaction_type == "focus"]
                completion_rates = [
                    i.metadata.get("completion_rate", 0) 
                    for i in interactions 
                    if "completion_rate" in (i.metadata or {})
                ]

                pattern = {
                    "focus_duration_avg": np.mean(focus_durations) if focus_durations else 0,
                    "completion_rate": np.mean(completion_rates) if completion_rates else 0,
                    "interaction_frequency": len(interactions),
                    "preferred_content_types": self._analyze_content_preferences(interactions)
                }

                learning_pattern = LearningPattern(
                    user_id=user_id,
                    preferred_style=self._determine_learning_style(pattern),
                    focus_duration_avg=pattern["focus_duration_avg"],
                    completion_rate=pattern["completion_rate"],
                    metrics=pattern
                )
                session.merge(learning_pattern)
                session.commit()
                logger.info(f"Learning pattern analysis completed for user {user_id}")
                return pattern
        except Exception as e:
            logger.error(f"Error analyzing learning pattern: {str(e)}")
            raise

    def _determine_learning_style(self, pattern: Dict) -> str:
        """Determine learning style based on interaction patterns."""
        if pattern["focus_duration_avg"] > 300:  # 5 minutes
            return "deep_focus"
        elif pattern["interaction_frequency"] > 50:
            return "active_learner"
        return "interactive"

    def _analyze_content_preferences(self, interactions: List[UserInteraction]) -> Dict[str, float]:
        """Analyze content type preferences."""
        type_durations = {}
        for interaction in interactions:
            content_type = interaction.metadata.get("content_type", "unknown")
            type_durations[content_type] = type_durations.get(content_type, 0) + interaction.duration

        total_duration = sum(type_durations.values())
        return {k: v/total_duration for k, v in type_durations.items()} if total_duration else {}

    async def process_audio_content(self, audio_file_path: str) -> Dict[str, Any]:
        """Process audio content using AssemblyAI."""
        try:
            logger.info(f"Processing audio file: {audio_file_path}")
            transcriber = aai.Transcriber()
            transcript = await transcriber.transcribe(audio_file_path)

            result = {
                "text": transcript.text,
                "confidence": transcript.confidence,
                "words": [
                    {
                        "text": word.text,
                        "start": word.start,
                        "end": word.end,
                        "confidence": word.confidence
                    }
                    for word in transcript.words
                ]
            }
            logger.info("Audio processing completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return {"error": str(e)}

    def adapt_content(
        self,
        content: str,
        user_id: str,
        complexity: str = "medium"
    ) -> Dict[str, Any]:
        """Adapt content based on user's learning pattern."""
        try:
            logger.info(f"Adapting content for user {user_id}")
            with self.DBSession() as session:
                pattern = session.query(LearningPattern).filter_by(user_id=user_id).first()

                if not pattern:
                    logger.warning(f"No learning pattern found for user {user_id}, using default adaptation")
                    return self._default_content_adaptation(content, complexity)

                if pattern.preferred_style == "deep_focus":
                    result = self._adapt_for_deep_focus(content)
                elif pattern.preferred_style == "active_learner":
                    result = self._adapt_for_active_learner(content)
                else:
                    result = self._adapt_for_interactive(content)

                logger.info("Content adaptation completed successfully")
                return result
        except Exception as e:
            logger.error(f"Error adapting content: {str(e)}")
            raise

    def _default_content_adaptation(self, content: str, complexity: str) -> Dict[str, Any]:
        """Default content adaptation strategy."""
        paragraphs = content.split('\n\n')
        if complexity == "simple":
            adapted = [p.split('. ')[0] + '.' for p in paragraphs]
        else:
            adapted = paragraphs

        return {
            "adapted_content": '\n\n'.join(adapted),
            "complexity": complexity,
            "structure": "default"
        }

    def _adapt_for_deep_focus(self, content: str) -> Dict[str, Any]:
        """Adapt content for deep focus learners."""
        sections = content.split('\n\n')
        adapted = []
        for section in sections:
            adapted.extend([
                "📚 " + section,
                "🔑 Key Points:",
                "• " + "\n• ".join(self._extract_key_points(section))
            ])

        return {
            "adapted_content": '\n\n'.join(adapted),
            "structure": "detailed",
            "includes_summary": True
        }

    def _adapt_for_active_learner(self, content: str) -> Dict[str, Any]:
        """Adapt content for active learners."""
        sections = content.split('\n\n')
        adapted = []
        for i, section in enumerate(sections, 1):
            adapted.extend([
                f"Step {i}: {section}",
                "✍️ Practice:",
                self._generate_quick_exercise(section)
            ])

        return {
            "adapted_content": '\n\n'.join(adapted),
            "structure": "interactive",
            "includes_exercises": True
        }

    def _adapt_for_interactive(self, content: str) -> Dict[str, Any]:
        """Adapt content for interactive learners."""
        sections = content.split('\n\n')
        adapted = []
        for section in sections:
            adapted.extend([
                "💡 " + section,
                "🤔 Think About:",
                self._generate_reflection_prompt(section)
            ])

        return {
            "adapted_content": '\n\n'.join(adapted),
            "structure": "reflective",
            "includes_prompts": True
        }

    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text."""
        sentences = text.split('. ')
        return [s for s in sentences if any(kw in s.lower() for kw in ['key', 'important', 'must', 'should', 'critical'])]

    def _generate_quick_exercise(self, text: str) -> str:
        """Generate a quick exercise based on content."""
        return f"Try to explain this concept in your own words: {text.split('.')[0]}"

    def _generate_reflection_prompt(self, text: str) -> str:
        """Generate a reflection prompt based on content."""
        return f"How would you apply this in your daily life: {text.split('.')[0]}?"

    def store_content_vector(self, content_id: str, content: str, content_type: str) -> None:
        """Store content vector in Pinecone and database."""
        try:
            logger.info(f"Storing content vector for content_id: {content_id}")
            # Generate vector embedding (simplified version)
            vector = np.random.rand(1536).tolist()  # In practice, use proper embedding model

            # Store in Pinecone
            self.vector_index.upsert([(content_id, vector, {"content_type": content_type})])

            # Store in database
            with self.DBSession() as session:
                content_vector = ContentVector(
                    content_id=content_id,
                    vector=vector,
                    content_type=content_type,
                    metadata={"original_content": content}
                )
                session.merge(content_vector)
                session.commit()
                logger.info("Content vector stored successfully")
        except Exception as e:
            logger.error(f"Error storing content vector: {str(e)}")
            raise

    def find_similar_content(self, content_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar content using vector similarity."""
        try:
            logger.info(f"Finding similar content for content_id: {content_id}")
            with self.DBSession() as session:
                content_vector = session.query(ContentVector).filter_by(content_id=content_id).first()

                if not content_vector:
                    logger.warning(f"No content vector found for content_id: {content_id}")
                    return []

                # Query Pinecone
                results = self.vector_index.query(
                    vector=content_vector.vector,
                    top_k=top_k,
                    include_metadata=True
                )

                logger.info(f"Found {len(results.matches)} similar content items")
                return [
                    {
                        "content_id": match.id,
                        "score": match.score,
                        "content_type": match.metadata.get("content_type")
                    }
                    for match in results.matches
                ]
        except Exception as e:
            logger.error(f"Error finding similar content: {str(e)}")
            raise

# Create singleton instance
ai_model = AdvancedAIModel()