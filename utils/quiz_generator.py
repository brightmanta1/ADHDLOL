import os
from openai import OpenAI

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_quiz(content):
    """Generate quiz questions from content using OpenAI."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Generate 3 multiple choice questions based on the content. "
                    "Return in JSON format with questions, options, and correct answers."
                },
                {"role": "user", "content": content}
            ],
            response_format={"type": "json_object"}
        )
        
        quiz_data = response.choices[0].message.content
        return [
            {
                "question": q["question"],
                "options": q["options"],
                "correct_answer": q["correct_answer"]
            }
            for q in quiz_data["questions"]
        ]
    except Exception as e:
        return [{"question": f"Error generating quiz: {str(e)}", "options": [], "correct_answer": ""}]
