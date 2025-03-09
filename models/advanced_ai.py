import os
import assemblyai as aai
from datetime import datetime, time, timedelta
import numpy as np
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
import json
import logging
import requests
from urllib.parse import urljoin
from collections import defaultdict
from pinecone import Pinecone, ServerlessSpec

from .database import init_db, UserInteraction, LearningPattern, ContentVector, UserSchedule
from .text_processor import TextProcessor  # Add TextProcessor import
from utils.visualization import ContentVisualizer # Add import at the top of the file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAIModel:
    def __init__(self):
        logger.info("Initializing AdvancedAIModel...")
        try:
            # Initialize Pinecone with new client
            self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            self.index_name = "adhd-learning-content"

            # Create index if it doesn't exist
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
            self.vector_index = self.pc.Index(self.index_name)

            # Initialize AssemblyAI
            aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')

            # Initialize database
            self.DBSession = init_db()
            # Initialize content visualizer
            self.visualizer = ContentVisualizer()

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
        meta_data: Dict = None
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
                    meta_data=meta_data or {}
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
                    i.meta_data.get("completion_rate", 0)
                    for i in interactions
                    if "completion_rate" in (i.meta_data or {})
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
            content_type = interaction.meta_data.get("content_type", "unknown")
            type_durations[content_type] = type_durations.get(content_type, 0) + interaction.duration

        total_duration = sum(type_durations.values())
        return {k: v / total_duration for k, v in type_durations.items()} if total_duration else {}

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
        """Adapt content based on user's learning pattern with enhanced hierarchy and visualizations."""
        try:
            logger.info(f"Adapting content for user {user_id}")
            with self.DBSession() as session:
                pattern = session.query(LearningPattern).filter_by(user_id=user_id).first()

                # Create text processor instance
                text_processor = TextProcessor()
                processed_text = text_processor.process_text(content, complexity)

                # Generate base content structure
                adapted_content = self._get_adapted_content(pattern, processed_text, complexity)

                # Generate visualizations
                try:
                    mindmap_svg = self.visualizer.generate_mindmap(
                        processed_text.topics,
                        processed_text.highlighted_terms
                    )
                    hierarchy_svg = self.visualizer.create_concept_hierarchy(
                        processed_text.topics,
                        processed_text.highlighted_terms
                    )

                    # Generate learning stats
                    content_stats = {
                        'completion_rate': pattern.completion_rate if pattern else 0.5,
                        'engagement_score': 0.75,  # Example value
                        'understanding_level': 0.8  # Example value
                    }
                    infographic_svg = self.visualizer.generate_learning_infographic(content_stats)

                    # Add visualizations to the response
                    adapted_content.update({
                        'visualizations': {
                            'mindmap': mindmap_svg,
                            'hierarchy': hierarchy_svg,
                            'infographic': infographic_svg
                        }
                    })
                except Exception as viz_error:
                    logger.error(f"Error generating visualizations: {str(viz_error)}")
                    adapted_content['visualization_error'] = str(viz_error)

                logger.info("Content adaptation completed successfully")
                return adapted_content

        except Exception as e:
            logger.error(f"Error adapting content: {str(e)}")
            raise

    def _get_adapted_content(self, pattern, processed_text, complexity):
        # Base adaptation
        if not pattern:
            logger.warning(f"No learning pattern found, using default adaptation")
            return {
                "content": processed_text.simplified,
                "topics": processed_text.topics,
                "highlighted_terms": processed_text.highlighted_terms,
                "tags": processed_text.tags,
                "key_concepts": processed_text.key_concepts,
                "complexity_score": processed_text.complexity_score
            }

        # Enhanced adaptation based on learning style
        if pattern.preferred_style == "deep_focus":
            return self._adapt_for_deep_focus(
                processed_text.simplified,
                processed_text.topics,
                processed_text.highlighted_terms
            )
        elif pattern.preferred_style == "active_learner":
            return self._adapt_for_active_learner(
                processed_text.simplified,
                processed_text.topics,
                processed_text.highlighted_terms
            )
        else:
            return self._adapt_for_interactive(
                processed_text.simplified,
                processed_text.topics,
                processed_text.highlighted_terms
            )


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

    def _adapt_for_deep_focus(self, content: str, topics: Dict[str, List[str]], highlighted_terms: Dict[str, str]) -> Dict[str, Any]:
        """Adapt content for deep focus learners with enhanced hierarchy."""
        formatted_sections = []

        # Process each topic
        for topic, subtopics in topics.items():
            topic_section = [
                f"\n📚 {topic}",  # Topic header with emoji
                "\n🔑 Key Terms:",
                *[f"• {term}" for term, color in highlighted_terms.items() if term in content],
                "\n📝 Details:",
                *[f"• {subtopic}" for subtopic in subtopics],
                "\n🎯 Key Points:",
                self._extract_key_points(content),
                "\n---"  # Visual separator
            ]
            formatted_sections.extend(topic_section)

        # Compose the final content
        final_content = "\n".join(formatted_sections)

        return {
            "content": final_content,
            "structure": "detailed",
            "topics": topics,
            "highlighted_terms": highlighted_terms
        }

    def _adapt_for_active_learner(self, content: str, topics: Dict[str, List[str]], highlighted_terms: Dict[str, str]) -> Dict[str, Any]:
        """Adapt content for active learners with enhanced hierarchy."""
        sections = []

        for topic, subtopics in topics.items():
            sections.extend([
                f"🎯 Topic: {topic}",
                "",
                "📌 Quick Overview:",
                *[f"• {subtopic}" for subtopic in subtopics],
                "",
                "✍️ Practice Points:",
                *[f"• Apply {term}" for term in highlighted_terms.keys()],
                "",
                "🤔 Exercise:",
                self._generate_quick_exercise(topic)
            ])

        return {
            "content": '\n'.join(sections),
            "structure": "interactive",
            "topics": topics,
            "highlighted_terms": highlighted_terms
        }

    def _adapt_for_interactive(self, content: str, topics: Dict[str, List[str]], highlighted_terms: Dict[str, str]) -> Dict[str, Any]:
        """Adapt content for interactive learners with enhanced hierarchy."""
        sections = []

        for topic, subtopics in topics.items():
            sections.extend([
                f"💡 Exploring: {topic}",
                "",
                "🎨 Visual Map:",
                *[f"• {subtopic}" for subtopic in subtopics],
                "",
                "🔍 Key Terms:",
                *[f"• {term} ({color})" for term, color in highlighted_terms.items()],
                "",
                "🤔 Think About:",
                self._generate_reflection_prompt(topic)
            ])

        return {
            "content": '\n'.join(sections),
            "structure": "reflective",
            "topics": topics,
            "highlighted_terms": highlighted_terms
        }


    def _extract_key_points(self, text: str) -> str:
        """Extract key points from text."""
        sentences = text.split('. ')
        key_sentences = [s for s in sentences if any(kw in s.lower() for kw in ['key', 'important', 'must', 'should', 'critical'])]
        return ". ".join(key_sentences) if key_sentences else "No key points found."

    def _generate_quick_exercise(self, text: str) -> str:
        """Generate a quick exercise based on content."""
        return f"Try to explain this concept in your own words: {text}"

    def _generate_reflection_prompt(self, text: str) -> str:
        """Generate a reflection prompt based on content."""
        return f"How would you apply this in your daily life: {text}?"

    def store_content_vector(
        self,
        content_id: str,
        content: str,
        content_type: str,
        video_meta_data: Dict = None
    ) -> None:
        """Store content vector in Pinecone and database."""
        try:
            logger.info(f"Storing content vector for content_id: {content_id}")
            # Generate vector embedding (simplified version)
            vector = np.random.rand(1536).tolist()  # In practice, use proper embedding model

            # Store in Pinecone
            meta_data = {"content_type": content_type}
            if video_meta_data:
                meta_data["video_meta_data"] = video_meta_data
            self.vector_index.upsert([(content_id, vector, meta_data)])

            # Store in database
            with self.DBSession() as session:
                content_vector = ContentVector(
                    content_id=content_id,
                    vector=vector,
                    content_type=content_type,
                    meta_data={"original_content": content, "video_meta_data": video_meta_data}
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

    async def process_video_content(self, video_url: str) -> Dict[str, Any]:
        """Process video content using Loom API and AssemblyAI."""
        try:
            logger.info(f"Processing video from Loom: {video_url}")

            # Extract video metadata from Loom
            video_metadata = await self._get_loom_metadata(video_url)

            # Get audio track from video for transcription
            audio_url = video_metadata.get('audio_url')
            if not audio_url:
                raise ValueError("No audio track found in video")

            # Transcribe audio using AssemblyAI
            transcriber = aai.Transcriber()
            transcript = await transcriber.transcribe(audio_url)

            # Create video summary
            summary = await self._generate_video_summary(transcript.text, video_metadata)

            # Store video content vector
            self.store_content_vector(
                content_id=f"video_{video_metadata['id']}",
                content=summary['text'],
                content_type="video",
                video_meta_data=summary
            )

            return {
                "summary": summary['text'],
                "timestamps": summary['timestamps'],
                "duration": video_metadata['duration'],
                "transcript": transcript.text
            }

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return {"error": str(e)}

    async def _get_loom_metadata(self, video_url: str) -> Dict[str, Any]:
        """Get video metadata from Loom API."""
        try:
            # Extract video ID from URL
            video_id = video_url.split('/')[-1]

            # Loom API endpoint
            api_url = f"https://api.loom.com/v1/videos/{video_id}"

            headers = {
                "Authorization": f"Bearer {os.getenv('LOOM_API_KEY')}",
                "Content-Type": "application/json"
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Error fetching Loom metadata: {str(e)}")
            raise

    async def _generate_video_summary(
        self,
        transcript: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a video summary with timestamps."""
        try:
            # Split transcript into segments
            segments = self._split_into_segments(transcript)

            # Generate summary for each segment
            timestamps = []
            summary_parts = []

            for i, segment in enumerate(segments):
                summary_part = self._summarize_segment(segment)
                timestamp = (i * metadata['duration']) / len(segments)

                timestamps.append({
                    "time": timestamp,
                    "text": summary_part
                })
                summary_parts.append(summary_part)

            return {
                "text": " ".join(summary_parts),
                "timestamps": timestamps,
                "duration": metadata['duration']
            }

        except Exception as e:
            logger.error(f"Error generating video summary: {str(e)}")
            raise

    def _split_into_segments(self, text: str, max_segments: int = 6) -> List[str]:
        """Split transcript into segments for summarization."""
        # Split into sentences
        sentences = text.split('. ')

        # Calculate segments
        segment_size = max(1, len(sentences) // max_segments)

        # Create segments
        segments = [
            '. '.join(sentences[i:i + segment_size])
            for i in range(0, len(sentences), segment_size)
        ]

        return segments[:max_segments]

    def _summarize_segment(self, text: str) -> str:
        """Summarize a segment of text."""
        # Extract key sentences based on importance
        sentences = text.split('. ')
        if len(sentences) <= 2:
            return text

        # Simple extractive summarization
        # In practice, you might want to use a more sophisticated approach
        return '. '.join(sentences[:2]) + '.'

    def analyze_effectiveness(self, user_id: str) -> Dict[str, Any]:
        """Analyze when a user is most effective based on interaction data."""
        try:
            logger.info(f"Analyzing effectiveness for user {user_id}")
            with self.DBSession() as session:
                # Get all user interactions
                interactions = session.query(UserInteraction).filter_by(user_id=user_id).all()

                if not interactions:
                    return {"error": "No interaction data found"}

                # Analyze effectiveness by hour
                hourly_effectiveness = defaultdict(list)
                for interaction in interactions:
                    hour = interaction.timestamp.hour
                    effectiveness = self._calculate_effectiveness_score(interaction)
                    hourly_effectiveness[hour].append(effectiveness)

                # Calculate average effectiveness for each hour
                peak_hours = {
                    hour: np.mean(scores)
                    for hour, scores in hourly_effectiveness.items()
                }

                # Identify top 3 most effective hours
                best_hours = sorted(
                    peak_hours.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]

                # Update learning pattern with peak hours
                pattern = session.query(LearningPattern).filter_by(user_id=user_id).first()
                if pattern:
                    pattern.peak_hours = {
                        "best_hours": best_hours,
                        "hourly_scores": peak_hours
                    }
                    session.commit()

                return {
                    "peak_hours": best_hours,
                    "hourly_effectiveness": peak_hours
                }

        except Exception as e:
            logger.error(f"Error analyzing effectiveness: {str(e)}")
            raise

    def _calculate_effectiveness_score(self, interaction: UserInteraction) -> float:
        """Calculate effectiveness score based on interaction metrics."""
        try:
            base_score = 0.0

            # Consider focus duration
            if interaction.duration > 0:
                base_score += min(interaction.duration / 3600, 1.0) * 0.4  # Up to 40% based on duration

            # Consider completion rate from metadata
            completion_rate = interaction.meta_data.get("completion_rate", 0)
            base_score += completion_rate * 0.3  # Up to 30% based on completion

            # Consider interaction type
            interaction_weights = {
                "focus": 0.3,
                "quiz_completion": 0.2,
                "content_creation": 0.15,
                "scroll": 0.05
            }
            base_score += interaction_weights.get(interaction.interaction_type, 0.1)

            return min(base_score, 1.0)  # Normalize to [0, 1]

        except Exception as e:
            logger.error(f"Error calculating effectiveness score: {str(e)}")
            return 0.0

    def generate_schedule(
        self,
        user_id: str,
        preferred_start_time: time = time(9, 0),  # 9:00 AM default
        preferred_end_time: time = time(17, 0)    # 5:00 PM default
    ) -> Dict[str, Any]:
        """Generate a personalized learning schedule based on effectiveness analysis."""
        try:
            logger.info(f"Generating schedule for user {user_id}")

            # Analyze effectiveness first
            effectiveness_data = self.analyze_effectiveness(user_id)
            if "error" in effectiveness_data:
                return effectiveness_data

            peak_hours = effectiveness_data["peak_hours"]

            # Create schedule blocks
            schedule_blocks = []
            current_time = preferred_start_time

            while current_time < preferred_end_time:
                block_hour = current_time.hour

                # Find effectiveness score for this hour
                hour_score = dict(peak_hours).get(block_hour, 0.5)

                # Determine activity type based on effectiveness
                if hour_score > 0.7:
                    activity_type = "deep_focus"
                    duration = 45  # 45 minutes for high effectiveness periods
                elif hour_score > 0.4:
                    activity_type = "interactive_learning"
                    duration = 30
                else:
                    activity_type = "review_and_practice"
                    duration = 20

                # Create schedule block
                block = {
                    "start_time": current_time.strftime("%H:%M"),
                    "duration": duration,
                    "activity_type": activity_type,
                    "effectiveness_score": hour_score
                }
                schedule_blocks.append(block)

                # Add break after each block
                break_duration = 15 if activity_type == "deep_focus" else 10
                current_time = (
                    datetime.combine(datetime.today(), current_time) +
                    timedelta(minutes=duration + break_duration)
                ).time()

            # Store schedule in database
            with self.DBSession() as session:
                schedule = UserSchedule(
                    user_id=user_id,
                    schedule_data=schedule_blocks,
                    start_time=preferred_start_time,
                    end_time=preferred_end_time,
                    effectiveness_metrics=effectiveness_data
                )
                session.add(schedule)
                session.commit()

            return {
                "schedule": schedule_blocks,
                "peak_performance_times": peak_hours,
                "recommendations": self._generate_schedule_recommendations(effectiveness_data)
            }

        except Exception as e:
            logger.error(f"Error generating schedule: {str(e)}")
            raise

    def _generate_schedule_recommendations(self, effectiveness_data: Dict) -> List[str]:
        """Generate recommendations based on effectiveness analysis."""
        recommendations = []
        peak_hours = effectiveness_data["peak_hours"]

        # Add time-based recommendations
        best_hour = peak_hours[0][0] if peak_hours else 9
        recommendations.append(
            f"Your peak performance time is around {best_hour:02d}:00. "
            "Schedule challenging tasks during this period."
        )

        # Add pattern-based recommendations
        if len(peak_hours) >= 3:
            morning_peak = any(hour < 12 for hour, _ in peak_hours)
            afternoon_peak = any(12 <= hour < 17 for hour, _ in peak_hours)
            evening_peak = any(hour >= 17 for hour, _ in peak_hours)

            if morning_peak:
                recommendations.append(
                    "You show strong morning performance. "
                    "Consider starting your day with complex learning tasks."
                )
            if afternoon_peak:
                recommendations.append(
                    "Afternoon sessions are productive for you. "
                    "Use this time for interactive learning activities."
                )
            if evening_peak:
                recommendations.append(
                    "You demonstrate good evening focus. "
                    "Plan revision and practice sessions for these hours."
                )

        return recommendations


# Create singleton instance
ai_model = AdvancedAIModel()