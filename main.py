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
    create_personalized_quiz
)

# Page configuration
st.set_page_config(
    page_title="ADHD Learning Platform",
    page_icon="🧠",
    layout="wide"
)

# Load custom CSS
with open('styles/custom.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
if 'current_module' not in st.session_state:
    st.session_state.current_module = 0
if 'pomodoro_active' not in st.session_state:
    st.session_state.pomodoro_active = False
if 'progress' not in st.session_state:
    st.session_state.progress = {}
if 'learning_style' not in st.session_state:
    st.session_state.learning_style = 'visual'

# Sidebar
with st.sidebar:
    st.title("Learning Settings")

    # Text display settings
    st.subheader("Display Settings")
    font_size = st.slider("Font Size", 12, 24, 16)
    high_contrast = st.toggle("High Contrast Mode", False)
    complexity = st.select_slider(
        "Content Complexity",
        options=["simple", "medium", "advanced"],
        value="medium"
    )

    # Focus tools
    st.subheader("Focus Tools")
    if st.button("Start Pomodoro" if not st.session_state.pomodoro_active else "Stop Pomodoro"):
        st.session_state.pomodoro_active = not st.session_state.pomodoro_active

    # Progress overview
    st.subheader("Your Progress")
    progress_chart = st.progress(sum(st.session_state.progress.values()) / 100 if st.session_state.progress else 0)

# Main content
st.title("ADHD-Optimized Learning Platform")

# Module selection
modules = [
    "Introduction to ADHD",
    "Focus Techniques",
    "Time Management",
    "Organization Skills"
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
    if 'error' not in processed_content:
        displayed_content = processed_content.get('content', content)
    else:
        displayed_content = content
        st.error("Content processing error. Showing original content.")

    # Display options
    display_mode = st.radio("Display Mode", ["Text", "Audio", "Both"])

    if display_mode in ["Text", "Both"]:
        st.markdown(f"""
        <div class="content-text" style="font-size: {font_size}px">
            {displayed_content}
        </div>
        """, unsafe_allow_html=True)

    if display_mode in ["Audio", "Both"]:
        audio_file = text_to_speech(displayed_content)
        st.audio(audio_file)

# Interactive elements
with st.container():
    st.subheader("Practice Quiz")
    if st.button("Generate Quiz"):
        quiz = create_personalized_quiz(displayed_content, complexity)
        if 'error' not in quiz:
            for i, question in enumerate(quiz.get('questions', [])):
                st.write(f"Q{i+1}: {question['question']}")
                answer = st.radio(
                    f"Select answer for question {i+1}:",
                    question['options'],
                    key=f"quiz_{i}"
                )
                if st.button(f"Check Answer {i+1}", key=f"check_{i}"):
                    if answer == question['correct_answer']:
                        st.success("Correct! 🎉")
                        update_progress(selected_module, 10)
                    else:
                        st.error("Try again!")
        else:
            st.error("Error generating quiz. Please try again.")

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
        "complexity_preference": complexity
    }
    analysis = analyze_learning_style(user_data)
    if 'error' not in analysis:
        st.session_state.learning_style = analysis.get('preferred_style', 'visual')
        st.sidebar.info(f"Recommended learning style: {st.session_state.learning_style}")
    else:
        st.sidebar.error("Could not analyze learning style")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for ADHD learners")