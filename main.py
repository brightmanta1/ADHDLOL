import streamlit as st
import time
from datetime import datetime
import json
from utils.text_processing import text_to_speech
from utils.focus_tools import PomodoroTimer, track_attention
from utils.progress_tracker import update_progress, get_badges
from utils.ai_models import (
    simplify_content,
    analyze_learning_style,
    generate_adaptive_content,
    create_personalized_quiz,
)
from utils.ai_models import ai_model  # Import ai_model
from utils.content_converter import content_converter  # Added import
import tempfile  # Added import
import os  # Added import


# Page configuration
st.set_page_config(page_title="ADHD Learning Platform", page_icon="🧠", layout="wide")

# Load custom CSS
with open("styles/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state
if "current_module" not in st.session_state:
    st.session_state.current_module = 0
if "pomodoro_active" not in st.session_state:
    st.session_state.pomodoro_active = False
if "progress" not in st.session_state:
    st.session_state.progress = {}
if "learning_style" not in st.session_state:
    st.session_state.learning_style = "visual"

# Sidebar
with st.sidebar:
    st.title("Learning Settings")

    # Text display settings
    st.subheader("Display Settings")
    font_size = st.slider("Font Size", 12, 24, 16)
    high_contrast = st.toggle("High Contrast Mode", False)
    complexity = st.select_slider(
        "Content Complexity", options=["simple", "medium", "advanced"], value="medium"
    )

    # Focus tools
    st.subheader("Focus Tools")
    if st.button(
        "Start Pomodoro" if not st.session_state.pomodoro_active else "Stop Pomodoro"
    ):
        st.session_state.pomodoro_active = not st.session_state.pomodoro_active

    # Progress overview
    st.subheader("Your Progress")
    progress_chart = st.progress(
        sum(st.session_state.progress.values()) / 100
        if st.session_state.progress
        else 0
    )

# Main content
st.title("ADHD-Optimized Learning Platform")

# Module selection
modules = [
    "Introduction to ADHD",
    "Focus Techniques",
    "Time Management",
    "Organization Skills",
]

selected_module = st.selectbox("Select Module", modules)

# Content area
with st.container():
    # Module content
    content = """
    ADHD affects focus and attention. It can make learning challenging.
    But with the right strategies, you can succeed!

    Key points:
    1. Break tasks into smaller parts
    2. Use timer-based focus sessions
    3. Create a structured environment
    """

    # Process content using AI
    processed_content = simplify_content(content, complexity)
    if "error" not in processed_content:
        displayed_content = processed_content.get("content", content)
    else:
        displayed_content = content
        st.error("Content processing error. Showing original content.")

    # Display options
    display_mode = st.radio("Display Mode", ["Text", "Audio", "Both"])

    if display_mode in ["Text", "Both"]:
        st.markdown(
            f"""
        <div class="content-text" style="font-size: {font_size}px">
            {displayed_content}
        </div>
        """,
            unsafe_allow_html=True,
        )

    if display_mode in ["Audio", "Both"]:
        audio_file = text_to_speech(displayed_content)
        st.audio(audio_file)

# File upload section (added here)
with st.container():
    st.subheader("Import Learning Materials")
    uploaded_file = st.file_uploader(
        "Upload PDF, Word, PowerPoint, or text files",
        type=["pdf", "docx", "pptx", "txt"],
    )

    url_input = st.text_input(
        "Or enter a URL (web article or YouTube video)", placeholder="https://..."
    )

    if uploaded_file:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name

            # Convert content
            with st.spinner("Processing file..."):
                converted_content = content_converter.convert_content(file_path)

                # Update display content
                displayed_content = converted_content["content"]
                st.success("File processed successfully!")

                # Clean up temporary file
                os.unlink(file_path)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    elif url_input:
        try:
            with st.spinner("Processing URL..."):
                converted_content = content_converter.convert_content(url_input)
                displayed_content = converted_content["content"]
                st.success("URL content processed successfully!")

        except Exception as e:
            st.error(f"Error processing URL: {str(e)}")


# Interactive elements
with st.container():
    st.subheader("Practice Quiz")
    if st.button("Generate Quiz"):
        quiz = create_personalized_quiz(displayed_content, complexity)
        if "error" not in quiz:
            for i, question in enumerate(quiz.get("questions", [])):
                st.write(f"Q{i+1}: {question['question']}")
                answer = st.radio(
                    f"Select answer for question {i+1}:",
                    question["options"],
                    key=f"quiz_{i}",
                )
                if st.button(f"Check Answer {i+1}", key=f"check_{i}"):
                    if answer == question["correct_answer"]:
                        st.success("Correct! 🎉")
                        update_progress(selected_module, 10)
                    else:
                        st.error("Try again!")
        else:
            st.error("Error generating quiz. Please try again.")

# Visualization section
with st.container():
    st.subheader("Visual Learning Aids")

    # Get adapted content with visualizations
    adapted_content = ai_model.adapt_content(
        content=displayed_content,
        user_id="current_user",  # Replace with actual user ID when authentication is added
        complexity=complexity,
    )

    if "visualizations" in adapted_content:
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(
            ["📊 Mind Map", "🌳 Concept Hierarchy", "📈 Learning Progress"]
        )

        with viz_tab1:
            st.markdown(
                adapted_content["visualizations"]["mindmap"], unsafe_allow_html=True
            )
            st.caption("Mind map showing relationships between concepts")

        with viz_tab2:
            st.markdown(
                adapted_content["visualizations"]["hierarchy"], unsafe_allow_html=True
            )
            st.caption("Hierarchical view of topics and subtopics")

        with viz_tab3:
            st.markdown(
                adapted_content["visualizations"]["infographic"], unsafe_allow_html=True
            )
            st.caption("Your learning progress and engagement metrics")
    elif "visualization_error" in adapted_content:
        st.error(
            f"Could not generate visualizations: {adapted_content['visualization_error']}"
        )


# Focus tracking and reminders
if st.session_state.pomodoro_active:
    timer = PomodoroTimer()
    remaining_time = timer.get_remaining_time()
    st.sidebar.metric("Focus Timer", f"{remaining_time//60}:{remaining_time%60:02d}")

    if timer.should_take_break():
        st.balloons()
        st.warning("Time for a 5-minute break! Stand up and stretch! 🧘‍♂️")

# Achievements and badges
st.sidebar.subheader("Your Achievements")
badges = get_badges(st.session_state.progress)
for badge in badges:
    st.sidebar.markdown(f"🏆 {badge['name']}: {badge['description']}")

# Learning style analysis
if st.sidebar.button("Analyze Learning Style"):
    user_data = {
        "progress": st.session_state.progress,
        "completed_modules": len(st.session_state.progress),
        "preferred_display_mode": display_mode,
        "complexity_preference": complexity,
    }
    analysis = analyze_learning_style(user_data)
    if "error" not in analysis:
        st.session_state.learning_style = analysis.get("preferred_style", "visual")
        st.sidebar.info(
            f"Recommended learning style: {st.session_state.learning_style}"
        )
    else:
        st.sidebar.error("Could not analyze learning style")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for ADHD learners")
