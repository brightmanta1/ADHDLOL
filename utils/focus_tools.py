import time
from datetime import datetime, timedelta

class PomodoroTimer:
    def __init__(self, work_duration=25*60, break_duration=5*60):
        self.work_duration = work_duration
        self.break_duration = break_duration
        self.start_time = datetime.now()
        self.is_break = False
    
    def get_remaining_time(self):
        """Get remaining time in current session."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        current_duration = self.break_duration if self.is_break else self.work_duration
        remaining = max(0, current_duration - elapsed)
        
        if remaining == 0:
            self.is_break = not self.is_break
            self.start_time = datetime.now()
            
        return int(remaining)
    
    def should_take_break(self):
        """Check if it's time to take a break."""
        return self.get_remaining_time() == 0 and not self.is_break

def track_attention(session_start_time):
    """Track user attention based on interaction patterns."""
    current_time = time.time()
    session_duration = current_time - session_start_time
    
    # Simple attention tracking based on session duration
    if session_duration > 45 * 60:  # 45 minutes
        return "Consider taking a break"
    return None
