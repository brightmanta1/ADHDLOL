from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, ForeignKey, Time
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
from datetime import datetime, time

Base = declarative_base()

class UserInteraction(Base):
    __tablename__ = 'user_interactions'

    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    content_id = Column(String)
    interaction_type = Column(String)  # scroll, click, focus
    duration = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta_data = Column(JSON)  # Changed from metadata to meta_data
    effectiveness_score = Column(Float)  # Track effectiveness of each interaction

class LearningPattern(Base):
    __tablename__ = 'learning_patterns'

    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    preferred_style = Column(String)
    focus_duration_avg = Column(Float)
    completion_rate = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)
    metrics = Column(JSON)
    peak_hours = Column(JSON)  # Store peak performance hours

class ContentVector(Base):
    __tablename__ = 'content_vectors'

    id = Column(Integer, primary_key=True)
    content_id = Column(String, unique=True)
    vector = Column(JSON)  # Store vector embeddings
    content_type = Column(String)
    meta_data = Column(JSON)  # Changed from metadata to meta_data
    created_at = Column(DateTime, default=datetime.utcnow)

class UserSchedule(Base):  # Track user schedules
    __tablename__ = 'user_schedules'

    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    schedule_data = Column(JSON)  # Store detailed schedule
    start_time = Column(Time)
    end_time = Column(Time)
    created_at = Column(DateTime, default=datetime.utcnow)
    effectiveness_metrics = Column(JSON)  # Store effectiveness data for different time slots

# Database initialization
def init_db():
    engine = create_engine(os.getenv('DATABASE_URL'))
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)