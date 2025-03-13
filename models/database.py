from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    JSON,
    DateTime,
    ForeignKey,
    Time,
    event,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
import os
import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any, Union
import json
from pathlib import Path
from contextlib import contextmanager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

Base = declarative_base()


class UserInteraction(Base):
    __tablename__ = "user_interactions"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    content_id = Column(String, nullable=False)
    interaction_type = Column(String, nullable=False)  # scroll, click, focus
    duration = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)
    effectiveness_score = Column(Float)

    user = relationship("User", back_populates="interactions")


class LearningPattern(Base):
    __tablename__ = "learning_patterns"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    preferred_style = Column(String)
    focus_duration_avg = Column(Float)
    completion_rate = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)
    metrics = Column(JSON)
    peak_hours = Column(JSON)

    user = relationship("User", back_populates="learning_patterns")


class ContentVector(Base):
    __tablename__ = "content_vectors"

    id = Column(Integer, primary_key=True)
    content_id = Column(String, unique=True, nullable=False)
    vector = Column(JSON)
    content_type = Column(String, nullable=False)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserSchedule(Base):
    __tablename__ = "user_schedules"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    schedule_data = Column(JSON)
    start_time = Column(Time)
    end_time = Column(Time)
    created_at = Column(DateTime, default=datetime.utcnow)
    effectiveness_metrics = Column(JSON)

    user = relationship("User", back_populates="schedules")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

    interactions = relationship("UserInteraction", back_populates="user")
    learning_patterns = relationship("LearningPattern", back_populates="user")
    learning_sessions = relationship("LearningSession", back_populates="user")
    progress = relationship("Progress", back_populates="user")
    quiz_results = relationship("QuizResult", back_populates="user")
    schedules = relationship("UserSchedule", back_populates="user")


class LearningSession(Base):
    __tablename__ = "learning_sessions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    topic = Column(String, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    duration = Column(Integer)
    score = Column(Float)
    notes = Column(String)

    user = relationship("User", back_populates="learning_sessions")


class Progress(Base):
    __tablename__ = "progress"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    topic = Column(String, nullable=False)
    completed_items = Column(JSON, nullable=False)
    score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="progress")


class QuizResult(Base):
    __tablename__ = "quiz_results"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    quiz_id = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    answers = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="quiz_results")


@event.listens_for(Base.metadata, "after_create")
def create_defaults(target, connection, **kw):
    """Create default values after tables are created"""
    session = scoped_session(sessionmaker(bind=connection))()
    try:
        # Add any default data here if needed
        session.commit()
    except Exception as e:
        logger.error(f"Error creating defaults: {str(e)}")
        session.rollback()
    finally:
        session.close()


class Database:
    def __init__(self, db_path: Union[str, Path] = "./data/app.db"):
        """
        Инициализация подключения к базе данных

        Args:
            db_path: Путь к файлу базы данных
        """
        try:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Создаем строку подключения
            self.db_url = f"sqlite:///{self.db_path}"
            self.engine = create_engine(
                self.db_url, pool_pre_ping=True, pool_recycle=3600
            )
            self.Session = scoped_session(sessionmaker(bind=self.engine))

            # Создаем таблицы
            Base.metadata.create_all(self.engine)

            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    @contextmanager
    def session_scope(self):
        """Контекстный менеджер для сессии SQLAlchemy"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()

    def add_user(self, username: str, email: str, password_hash: str) -> Optional[int]:
        """
        Добавление нового пользователя

        Args:
            username: Имя пользователя
            email: Email
            password_hash: Хэш пароля

        Returns:
            Optional[int]: ID нового пользователя
        """
        try:
            with self.session_scope() as session:
                user = User(username=username, email=email, password_hash=password_hash)
                session.add(user)
                session.flush()
                return user.id

        except Exception as e:
            logger.error(f"Error adding user: {str(e)}")
            return None

    def get_user(self, user_id: int) -> Optional[Dict]:
        """
        Получение информации о пользователе

        Args:
            user_id: ID пользователя

        Returns:
            Optional[Dict]: Информация о пользователе
        """
        try:
            with self.session_scope() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return None

                return {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "created_at": user.created_at,
                    "last_login": user.last_login,
                }

        except Exception as e:
            logger.error(f"Error getting user: {str(e)}")
            return None

    def add_learning_session(
        self,
        user_id: int,
        topic: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        score: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> Optional[int]:
        """
        Добавление сессии обучения

        Args:
            user_id: ID пользователя
            topic: Тема
            start_time: Время начала
            end_time: Время окончания
            score: Оценка
            notes: Заметки

        Returns:
            Optional[int]: ID новой сессии
        """
        try:
            duration = None
            if end_time:
                duration = int((end_time - start_time).total_seconds())

            with self.session_scope() as session:
                learning_session = LearningSession(
                    user_id=user_id,
                    topic=topic,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    score=score,
                    notes=notes,
                )
                session.add(learning_session)
                session.flush()
                return learning_session.id

        except Exception as e:
            logger.error(f"Error adding learning session: {str(e)}")
            return None

    def update_progress(
        self,
        user_id: int,
        topic: str,
        completed_items: List[str],
        score: Optional[float] = None,
    ) -> Optional[int]:
        """
        Обновление прогресса обучения

        Args:
            user_id: ID пользователя
            topic: Тема
            completed_items: Завершенные элементы
            score: Оценка

        Returns:
            Optional[int]: ID записи прогресса
        """
        try:
            with self.session_scope() as session:
                progress = Progress(
                    user_id=user_id,
                    topic=topic,
                    completed_items=completed_items,
                    score=score,
                )
                session.add(progress)
                session.flush()
                return progress.id

        except Exception as e:
            logger.error(f"Error updating progress: {str(e)}")
            return None

    def add_quiz_result(
        self, user_id: int, quiz_id: str, score: float, answers: Dict[str, Any]
    ) -> Optional[int]:
        """
        Добавление результатов теста

        Args:
            user_id: ID пользователя
            quiz_id: ID теста
            score: Оценка
            answers: Ответы на вопросы

        Returns:
            Optional[int]: ID записи результата
        """
        try:
            with self.session_scope() as session:
                quiz_result = QuizResult(
                    user_id=user_id, quiz_id=quiz_id, score=score, answers=answers
                )
                session.add(quiz_result)
                session.flush()
                return quiz_result.id

        except Exception as e:
            logger.error(f"Error adding quiz result: {str(e)}")
            return None

    def get_user_progress(
        self, user_id: int, topic: Optional[str] = None
    ) -> List[Dict]:
        """
        Получение прогресса пользователя

        Args:
            user_id: ID пользователя
            topic: Тема (опционально)

        Returns:
            List[Dict]: Список записей прогресса
        """
        try:
            with self.session_scope() as session:
                query = session.query(Progress).filter(Progress.user_id == user_id)

                if topic:
                    query = query.filter(Progress.topic == topic)

                query = query.order_by(Progress.timestamp.desc())

                return [
                    {
                        "id": p.id,
                        "topic": p.topic,
                        "completed_items": p.completed_items,
                        "score": p.score,
                        "timestamp": p.timestamp,
                    }
                    for p in query.all()
                ]

        except Exception as e:
            logger.error(f"Error getting user progress: {str(e)}")
            return []

    def get_quiz_history(
        self, user_id: int, quiz_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Получение истории тестов

        Args:
            user_id: ID пользователя
            quiz_id: ID теста (опционально)

        Returns:
            List[Dict]: Список результатов тестов
        """
        try:
            with self.session_scope() as session:
                query = session.query(QuizResult).filter(QuizResult.user_id == user_id)

                if quiz_id:
                    query = query.filter(QuizResult.quiz_id == quiz_id)

                query = query.order_by(QuizResult.timestamp.desc())

                return [
                    {
                        "id": qr.id,
                        "quiz_id": qr.quiz_id,
                        "score": qr.score,
                        "answers": qr.answers,
                        "timestamp": qr.timestamp,
                    }
                    for qr in query.all()
                ]

        except Exception as e:
            logger.error(f"Error getting quiz history: {str(e)}")
            return []

    def cleanup(self):
        """Очистка старых данных"""
        try:
            with self.session_scope() as session:
                # Удаление старых сессий (старше 30 дней)
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)

                session.query(LearningSession).filter(
                    LearningSession.start_time < thirty_days_ago
                ).delete()

                # Удаление старых результатов тестов
                session.query(QuizResult).filter(
                    QuizResult.timestamp < thirty_days_ago
                ).delete()

                logger.info("Database cleanup completed")

        except Exception as e:
            logger.error(f"Error cleaning up database: {str(e)}")
            # Не вызываем исключение, чтобы не прерывать процесс очистки


# Инициализация базы данных
def init_db(db_url: Optional[str] = None) -> sessionmaker:
    """
    Инициализация базы данных для использования с SQLAlchemy

    Args:
        db_url: URL подключения к базе данных

    Returns:
        sessionmaker: Фабрика сессий SQLAlchemy
    """
    try:
        if db_url is None:
            db_url = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")

        engine = create_engine(db_url)
        Base.metadata.create_all(engine)
        return sessionmaker(bind=engine)
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise
