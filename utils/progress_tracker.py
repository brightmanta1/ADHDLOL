import streamlit as st

def update_progress(module, points):
    """Update progress for a specific module."""
    if module not in st.session_state.progress:
        st.session_state.progress[module] = 0
    st.session_state.progress[module] = min(100, st.session_state.progress[module] + points)

def get_badges(progress):
    """Get earned badges based on progress."""
    badges = []

    # Define badge thresholds
    if sum(progress.values()) >= 300:
        badges.append({
            "name": "Learning Master",
            "description": "Completed 3 modules with excellence!"
        })
    elif sum(progress.values()) >= 200:
        badges.append({
            "name": "Knowledge Seeker",
            "description": "Making great progress!"
        })
    elif sum(progress.values()) >= 100:
        badges.append({
            "name": "Focus Champion",
            "description": "Completed your first module!"
        })

    return badges