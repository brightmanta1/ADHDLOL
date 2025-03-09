import streamlit as st
import time
from datetime import datetime
import json
from utils.text_processing import simplify_text, text_to_speech
from utils.focus_tools import PomodoroTimer, track_attention
from utils.quiz_generator import generate_quiz
from utils.progress_tracker import update_progress, get_badges

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

# Sidebar
with st.sidebar:
    st.title("Learning Settings")
    
    # Text display settings
    st.subheader("Display Settings")
    font_size = st.slider("Font Size", 12, 24, 16)
    high_contrast = st.toggle("High Contrast Mode", False)
    
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
    # Module content (simplified example)
    content = """
    ADHD affects focus and attention. It can make learning challenging.
    But with the right strategies, you can succeed!
    
    Key points:
    1. Break tasks into smaller parts
    2. Use timer-based focus sessions
    3. Create a structured environment
    """
    
    # Text processing
    simplified_text = simplify_text(content)
    
    # Display options
    display_mode = st.radio("Display Mode", ["Text", "Audio", "Both"])
    
    if display_mode in ["Text", "Both"]:
        st.markdown(f"""
        <div class="content-text" style="font-size: {font_size}px">
            {simplified_text}
        </div>
        """, unsafe_allow_html=True)
    
    if display_mode in ["Audio", "Both"]:
        audio_file = text_to_speech(simplified_text)
        st.audio(audio_file)

# Interactive elements
with st.container():
    st.subheader("Practice Quiz")
    if st.button("Generate Quiz"):
        quiz = generate_quiz(content)
        for i, question in enumerate(quiz):
            st.write(f"Q{i+1}: {question['question']}")
            answer = st.radio(f"Select answer for question {i+1}:", question['options'])
            if st.button(f"Check Answer {i+1}"):
                if answer == question['correct_answer']:
                    st.success("Correct! 🎉")
                    update_progress(selected_module, 10)
                else:
                    st.error("Try again!")

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

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for ADHD learners")
