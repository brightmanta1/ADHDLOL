"""
Utility module for printing processing results from various sources (video, text, audio).
"""

from typing import Dict, Any, Optional
import json
from dataclasses import asdict
from enum import Enum, auto


class ContentType(Enum):
    VIDEO = auto()
    TEXT = auto()
    AUDIO = auto()
    IMAGE = auto()


def print_processing_results(
    result: Dict[str, Any], content_type: Optional[ContentType] = None
) -> None:
    """
    Print processing results in a structured format for any content type.

    Args:
        result: Dictionary containing processing results
        content_type: Type of content that was processed (optional, will be auto-detected if not provided)
    """
    # Auto-detect content type if not provided
    if not content_type:
        if "video_info" in result:
            content_type = ContentType.VIDEO
        elif "text_info" in result:
            content_type = ContentType.TEXT
        elif "audio_info" in result:
            content_type = ContentType.AUDIO
        elif "image_info" in result:
            content_type = ContentType.IMAGE

    print("\n" + "=" * 80)
    print(f"{content_type.name if content_type else 'PROCESSING'} RESULTS")
    print("=" * 80)

    # Print general information
    print(f"\nStatus: {result['status']}")

    # Print content-specific information
    if content_type == ContentType.VIDEO:
        print(f"Total duration: {result.get('duration', 0):.2f} seconds")
        print(f"Number of segments: {len(result.get('segments', []))}")
    elif content_type == ContentType.TEXT:
        print(f"Text length: {result.get('text_length', 0)} characters")
        print(f"Language: {result.get('language', 'unknown')}")
    elif content_type == ContentType.AUDIO:
        print(f"Audio duration: {result.get('duration', 0):.2f} seconds")
        print(f"Sample rate: {result.get('sample_rate', 'unknown')} Hz")
    elif content_type == ContentType.IMAGE:
        print(f"Image dimensions: {result.get('dimensions', 'unknown')}")
        print(f"Format: {result.get('format', 'unknown')}")

    # Print segments/chunks if available
    segments = result.get("segments", [])
    if segments:
        for i, segment in enumerate(segments, 1):
            print(f"\n{'='*50}")
            print(f"Segment/Chunk {i}:")

            # Time information for video/audio
            if content_type in [ContentType.VIDEO, ContentType.AUDIO]:
                print(
                    f"Time: {segment.get('start_time', 0):.2f}s - {segment.get('end_time', 0):.2f}s"
                )
                print(
                    f"Duration: {segment.get('end_time', 0) - segment.get('start_time', 0):.2f}s"
                )

            # Print key points/summary
            if "key_points" in segment:
                print("\nKey Points:")
                for point in segment["key_points"]:
                    print(f"- {point}")

            # Print questions if available
            if (
                "interactive_elements" in segment
                and "questions" in segment["interactive_elements"]
            ):
                print("\nQuestions:")
                for q in segment["interactive_elements"]["questions"]:
                    print(f"- {q['question']}")
                    print(f"  Type: {q['type']}")
                    print(f"  Difficulty: {q['difficulty']}")
                    if "options" in q:
                        print("  Options:")
                        for opt in q["options"]:
                            print(f"    * {opt}")
                    print(f"  Answer: {q['answer']}")

            # Print concepts/terms
            if "concepts" in segment:
                print("\nConcepts:")
                for concept in segment["concepts"]:
                    print(f"- {concept['term']}: {concept['definition']}")

            # Print tags
            if "tags" in segment:
                print("\nTags:")
                print(", ".join(segment["tags"]))

    # Print additional content-specific information
    if "metadata" in result:
        print("\nMetadata:")
        for key, value in result["metadata"].items():
            print(f"- {key}: {value}")

    print("\n" + "=" * 80)


def save_results_to_file(result: Dict[str, Any], filename: str) -> None:
    """
    Save processing results to a JSON file.

    Args:
        result: Dictionary containing processing results
        filename: Name of the file to save results to
    """
    # Convert dataclasses to dictionaries if present
    serializable_result = {}
    for key, value in result.items():
        if hasattr(value, "__dataclass_fields__"):
            serializable_result[key] = asdict(value)
        else:
            serializable_result[key] = value

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)
