"""
Модуль для отслеживания прогресса обучения.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from .cache_manager import CacheManager


@dataclass
class LearningProgress:
    """Структура данных для хранения прогресса обучения"""

    user_id: str
    topic: str
    start_date: datetime
    last_update: datetime
    completed_items: List[str]
    scores: Dict[str, float]
    time_spent: timedelta
    difficulty_level: float
    notes: List[str]


class ProgressTracker:
    def __init__(self):
        """Инициализация трекера прогресса"""
        self.cache_manager = CacheManager(cache_dir="./cache/progress")
        self.active_progress: Dict[str, LearningProgress] = {}

        logging.info("ProgressTracker initialized")

    def start_tracking(self, user_id: str, topic: str) -> LearningProgress:
        """
        Начало отслеживания прогресса

        Args:
            user_id: ID пользователя
            topic: Тема обучения

        Returns:
            LearningProgress: Информация о прогрессе
        """
        try:
            # Проверка существующего прогресса
            progress_key = f"{user_id}_{topic}"
            if progress_key in self.active_progress:
                return self.active_progress[progress_key]

            # Создание нового прогресса
            progress = LearningProgress(
                user_id=user_id,
                topic=topic,
                start_date=datetime.now(),
                last_update=datetime.now(),
                completed_items=[],
                scores={},
                time_spent=timedelta(),
                difficulty_level=1.0,
                notes=[],
            )

            self.active_progress[progress_key] = progress
            self._save_progress(progress)

            logging.info(
                f"Started tracking progress for user {user_id} on topic {topic}"
            )
            return progress

        except Exception as e:
            logging.error(f"Error starting progress tracking: {str(e)}")
            raise

    def update_progress(
        self,
        user_id: str,
        topic: str,
        completed_item: Optional[str] = None,
        score: Optional[float] = None,
        time_delta: Optional[timedelta] = None,
        note: Optional[str] = None,
    ) -> LearningProgress:
        """
        Обновление прогресса обучения

        Args:
            user_id: ID пользователя
            topic: Тема обучения
            completed_item: Завершенный элемент
            score: Оценка за элемент
            time_delta: Затраченное время
            note: Заметка

        Returns:
            LearningProgress: Обновленный прогресс
        """
        try:
            progress_key = f"{user_id}_{topic}"
            if progress_key not in self.active_progress:
                self.start_tracking(user_id, topic)

            progress = self.active_progress[progress_key]

            # Обновление данных
            if completed_item:
                progress.completed_items.append(completed_item)

            if score is not None:
                progress.scores[completed_item] = score

            if time_delta:
                progress.time_spent += time_delta

            if note:
                progress.notes.append(note)

            # Обновление времени
            progress.last_update = datetime.now()

            # Перерасчет сложности
            progress.difficulty_level = self._calculate_difficulty(progress)

            # Сохранение прогресса
            self._save_progress(progress)

            logging.info(f"Updated progress for user {user_id} on topic {topic}")
            return progress

        except Exception as e:
            logging.error(f"Error updating progress: {str(e)}")
            raise

    def get_progress(self, user_id: str, topic: str) -> Optional[LearningProgress]:
        """
        Получение информации о прогрессе

        Args:
            user_id: ID пользователя
            topic: Тема обучения

        Returns:
            Optional[LearningProgress]: Информация о прогрессе
        """
        try:
            progress_key = f"{user_id}_{topic}"

            # Проверка активного прогресса
            if progress_key in self.active_progress:
                return self.active_progress[progress_key]

            # Проверка кэша
            cached_progress = self.cache_manager.get(progress_key)
            if cached_progress:
                progress = self._deserialize_progress(cached_progress)
                self.active_progress[progress_key] = progress
                return progress

            return None

        except Exception as e:
            logging.error(f"Error getting progress: {str(e)}")
            raise

    def get_statistics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Получение статистики обучения

        Args:
            user_id: ID пользователя
            days: Количество дней для анализа

        Returns:
            Dict[str, Any]: Статистика обучения
        """
        try:
            start_date = datetime.now() - timedelta(days=days)

            # Сбор всего прогресса пользователя
            user_progress = [
                p
                for p in self.active_progress.values()
                if p.user_id == user_id and p.start_date >= start_date
            ]

            if not user_progress:
                return {
                    "total_topics": 0,
                    "total_completed_items": 0,
                    "average_score": 0.0,
                    "total_time_hours": 0.0,
                    "topics_breakdown": {},
                }

            # Расчет статистики
            total_completed = sum(len(p.completed_items) for p in user_progress)
            total_scores = sum(
                sum(scores.values()) for p in user_progress for scores in [p.scores]
            )
            total_items_with_scores = sum(len(p.scores) for p in user_progress)

            topics_breakdown = {}
            for progress in user_progress:
                topics_breakdown[progress.topic] = {
                    "completed_items": len(progress.completed_items),
                    "average_score": (
                        sum(progress.scores.values()) / len(progress.scores)
                        if progress.scores
                        else 0
                    ),
                    "time_spent_hours": progress.time_spent.total_seconds() / 3600,
                }

            return {
                "total_topics": len(user_progress),
                "total_completed_items": total_completed,
                "average_score": (
                    total_scores / total_items_with_scores
                    if total_items_with_scores > 0
                    else 0
                ),
                "total_time_hours": sum(
                    p.time_spent.total_seconds() for p in user_progress
                )
                / 3600,
                "topics_breakdown": topics_breakdown,
            }

        except Exception as e:
            logging.error(f"Error getting statistics: {str(e)}")
            raise

    def _calculate_difficulty(self, progress: LearningProgress) -> float:
        """
        Расчет уровня сложности на основе прогресса

        Args:
            progress: Информация о прогрессе

        Returns:
            float: Уровень сложности (0.1-2.0)
        """
        try:
            if not progress.scores:
                return 1.0

            # Средний балл
            avg_score = sum(progress.scores.values()) / len(progress.scores)

            # Корректировка сложности
            if avg_score >= 0.9:
                return min(progress.difficulty_level * 1.2, 2.0)
            elif avg_score <= 0.6:
                return max(progress.difficulty_level * 0.8, 0.1)

            return progress.difficulty_level

        except Exception as e:
            logging.error(f"Error calculating difficulty: {str(e)}")
            return 1.0

    def _save_progress(self, progress: LearningProgress):
        """
        Сохранение прогресса в кэш

        Args:
            progress: Прогресс для сохранения
        """
        try:
            progress_data = {
                "user_id": progress.user_id,
                "topic": progress.topic,
                "start_date": progress.start_date.isoformat(),
                "last_update": progress.last_update.isoformat(),
                "completed_items": progress.completed_items,
                "scores": progress.scores,
                "time_spent": str(progress.time_spent),
                "difficulty_level": progress.difficulty_level,
                "notes": progress.notes,
            }

            progress_key = f"{progress.user_id}_{progress.topic}"
            self.cache_manager.set(progress_key, progress_data)

        except Exception as e:
            logging.error(f"Error saving progress: {str(e)}")
            raise

    def _deserialize_progress(self, data: Dict) -> LearningProgress:
        """
        Десериализация прогресса из кэша

        Args:
            data: Сериализованные данные

        Returns:
            LearningProgress: Десериализованный прогресс
        """
        try:
            return LearningProgress(
                user_id=data["user_id"],
                topic=data["topic"],
                start_date=datetime.fromisoformat(data["start_date"]),
                last_update=datetime.fromisoformat(data["last_update"]),
                completed_items=data["completed_items"],
                scores=data["scores"],
                time_spent=timedelta(seconds=float(data["time_spent"].split()[-1])),
                difficulty_level=data["difficulty_level"],
                notes=data["notes"],
            )
        except Exception as e:
            logging.error(f"Error deserializing progress: {str(e)}")
            raise
