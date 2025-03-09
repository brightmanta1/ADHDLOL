from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
from datetime import datetime

Base = declarative_base()

class UserInteraction(Base):
    __tablename__ = 'user_interactions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    content_id = Column(String)
    interaction_type = Column(String)  # scroll, click, focus
    duration = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)

class LearningPattern(Base):
    __tablename__ = 'learning_patterns'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    preferred_style = Column(String)
    focus_duration_avg = Column(Float)
    completion_rate = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)
    metrics = Column(JSON)

class ContentVector(Base):
    __tablename__ = 'content_vectors'
    
    id = Column(Integer, primary_key=True)
    content_id = Column(String, unique=True)
    vector = Column(JSON)  # Store vector embeddings
    content_type = Column(String)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database initialization
def init_db():
    engine = create_engine(os.getenv('DATABASE_URL'))
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
