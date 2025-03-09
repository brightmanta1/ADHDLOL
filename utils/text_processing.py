import os
from gtts import gTTS
import tempfile
from openai import OpenAI

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def simplify_text(text):
    """Simplify complex text using OpenAI GPT-4."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Simplify the following text for ADHD readers. "
                    "Break into short paragraphs, use bullet points, "
                    "and highlight key concepts."
                },
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error simplifying text: {str(e)}"

def text_to_speech(text):
    """Convert text to speech using gTTS."""
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        return f"Error converting text to speech: {str(e)}"
