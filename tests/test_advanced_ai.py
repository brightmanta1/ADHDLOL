import os
import asyncio
import logging
from datetime import datetime, time
import json
from models.advanced_ai import AdvancedAIModel
from models.database import init_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_advanced_ai():
    """Test all core functionalities of the Advanced AI Model"""
    try:
        logger.info("Initializing Advanced AI Model testing...")
        ai_model = AdvancedAIModel()

        # Test 1: Track user interaction with effectiveness metrics
        logger.info("Testing user interaction tracking...")
        ai_model.track_user_interaction(
            user_id="test_user_1",
            content_id="test_content_1",
            interaction_type="focus",
            duration=300.0,
            metadata={
                "complexity": "medium",
                "content_type": "text",
                "completion_rate": 0.85
            }
        )

        # Test 2: Analyze effectiveness
        logger.info("Testing effectiveness analysis...")
        effectiveness = ai_model.analyze_effectiveness("test_user_1")
        logger.info(f"Effectiveness analysis result: {effectiveness}")

        # Test 3: Generate personalized schedule
        logger.info("Testing schedule generation...")
        schedule = ai_model.generate_schedule(
            user_id="test_user_1",
            preferred_start_time=time(9, 0),
            preferred_end_time=time(17, 0)
        )
        logger.info(f"Generated schedule: {schedule}")

        # Test 4: Learning pattern analysis
        logger.info("Testing learning pattern analysis...")
        pattern = ai_model.analyze_learning_pattern("test_user_1")
        logger.info(f"Learning pattern result: {pattern}")

        # Test 5: Content adaptation
        test_content = """
        ADHD affects focus and attention. It can make learning challenging.
        But with the right strategies, you can succeed!
        """
        logger.info("Testing content adaptation...")
        adapted_content = ai_model.adapt_content(
            content=test_content,
            user_id="test_user_1",
            complexity="medium"
        )
        logger.info(f"Adapted content structure: {adapted_content.keys()}")

        # Test 6: Vector storage and similarity search
        logger.info("Testing vector storage...")
        ai_model.store_content_vector(
            content_id="test_content_1",
            content=test_content,
            content_type="lesson"
        )

        similar_content = ai_model.find_similar_content("test_content_1", top_k=3)
        logger.info(f"Similar content results: {similar_content}")

        # Test 7: Audio processing (async)
        logger.info("Testing audio processing...")
        # Create a small test audio file
        sample_audio = "test_audio.wav"  # You would need to create this
        audio_result = await ai_model.process_audio_content(sample_audio)
        logger.info(f"Audio processing result: {audio_result}")

        # Test 8: Video processing (async)
        logger.info("Testing video processing...")
        test_video_url = "https://www.loom.com/share/test-video-id"
        video_result = await ai_model.process_video_content(test_video_url)
        logger.info(f"Video processing result: {video_result}")

        logger.info("All tests completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(test_advanced_ai())