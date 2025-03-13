import asyncio
from utils.content_converter import ContentConverter


async def test_converter():
    # Initialize converter without AI model for basic testing
    converter = ContentConverter()

    url = "https://postnauka.org/longreads/157269"

    try:
        result = await converter.convert(url, content_type="url")
        print("\n=== Converted Content ===\n")
        print(result["content"])
        print("\n=== Topics ===\n")
        print(result["topics"])
        print("\n=== Highlighted Terms ===\n")
        print(result["highlighted_terms"])
        print("\n=== Tags ===\n")
        print(result["tags"])
        print("\n=== Metadata ===\n")
        print(result["metadata"])
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_converter())
