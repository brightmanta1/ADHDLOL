"""
Модуль инструментов для улучшения фокусировки внимания.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from .cache_manager import CacheManager


@dataclass
class FocusSession:
    """Структура данных для хранения информации о сессии фокусировки"""

    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[timedelta]
    focus_score: float
    interruptions: List[Dict]
    notes: List[str]


class FocusTools:
    def __init__(self):
        """Инициализация инструментов фокусировки"""
        self.cache_manager = CacheManager(cache_dir="./cache/focus")
        self.current_session: Optional[FocusSession] = None
        self.sessions: List[FocusSession] = []

        logging.info("FocusTools initialized")

    def start_session(self) -> FocusSession:
        """
        Начало новой сессии фокусировки

        Returns:
            FocusSession: Информация о начатой сессии
        """
        try:
            if self.current_session:
                self.end_session()

            self.current_session = FocusSession(
                start_time=datetime.now(),
                end_time=None,
                duration=None,
                focus_score=0.0,
                interruptions=[],
                notes=[],
            )

            logging.info(
                f"Started new focus session at {self.current_session.start_time}"
            )
            return self.current_session

        except Exception as e:
            logging.error(f"Error starting focus session: {str(e)}")
            raise

    def end_session(self) -> Optional[FocusSession]:
        """
        Завершение текущей сессии фокусировки

        Returns:
            Optional[FocusSession]: Информация о завершенной сессии
        """
        try:
            if not self.current_session:
                logging.warning("No active focus session to end")
                return None

            self.current_session.end_time = datetime.now()
            self.current_session.duration = (
                self.current_session.end_time - self.current_session.start_time
            )

            # Расчет финального фокус-скора
            self.current_session.focus_score = self._calculate_focus_score(
                self.current_session
            )

            # Сохранение сессии
            self.sessions.append(self.current_session)
            self._save_session(self.current_session)

            completed_session = self.current_session
            self.current_session = None

            logging.info(
                f"Ended focus session with score {completed_session.focus_score}"
            )
            return completed_session

        except Exception as e:
            logging.error(f"Error ending focus session: {str(e)}")
            raise

    def add_interruption(self, interruption_type: str, description: str):
        """
        Добавление информации о прерывании фокуса

        Args:
            interruption_type: Тип прерывания
            description: Описание прерывания
        """
        try:
            if not self.current_session:
                raise ValueError("No active focus session")

            interruption = {
                "type": interruption_type,
                "description": description,
                "timestamp": datetime.now().isoformat(),
            }

            self.current_session.interruptions.append(interruption)
            logging.info(f"Added interruption: {interruption_type}")

        except Exception as e:
            logging.error(f"Error adding interruption: {str(e)}")
            raise

    def add_note(self, note: str):
        """
        Добавление заметки к текущей сессии

        Args:
            note: Текст заметки
        """
        try:
            if not self.current_session:
                raise ValueError("No active focus session")

            self.current_session.notes.append(note)
            logging.info("Added note to current session")

        except Exception as e:
            logging.error(f"Error adding note: {str(e)}")
            raise

    def get_focus_stats(self, days: int = 7) -> Dict:
        """
        Получение статистики фокусировки за период

        Args:
            days: Количество дней для анализа

        Returns:
            Dict: Статистика фокусировки
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            relevant_sessions = [s for s in self.sessions if s.start_time >= start_date]

            total_duration = sum(
                (s.duration for s in relevant_sessions if s.duration), timedelta()
            )

            avg_focus_score = (
                sum(s.focus_score for s in relevant_sessions) / len(relevant_sessions)
                if relevant_sessions
                else 0
            )

            interruptions_by_type = {}
            for session in relevant_sessions:
                for interruption in session.interruptions:
                    int_type = interruption["type"]
                    interruptions_by_type[int_type] = (
                        interruptions_by_type.get(int_type, 0) + 1
                    )

            return {
                "total_sessions": len(relevant_sessions),
                "total_duration_hours": total_duration.total_seconds() / 3600,
                "average_focus_score": avg_focus_score,
                "total_interruptions": sum(interruptions_by_type.values()),
                "interruptions_by_type": interruptions_by_type,
            }

        except Exception as e:
            logging.error(f"Error getting focus stats: {str(e)}")
            raise

    def _calculate_focus_score(self, session: FocusSession) -> float:
        """
        Расчет оценки фокусировки для сессии

        Args:
            session: Сессия для анализа

        Returns:
            float: Оценка фокусировки (0-1)
        """
        try:
            if not session.duration:
                return 0.0

            # Базовая оценка
            base_score = 1.0

            # Штраф за прерывания
            interruption_penalty = len(session.interruptions) * 0.1

            # Бонус за длительность (максимум 0.2)
            duration_hours = session.duration.total_seconds() / 3600
            duration_bonus = min(duration_hours * 0.1, 0.2)

            # Итоговая оценка
            final_score = base_score - interruption_penalty + duration_bonus

            return max(0.0, min(1.0, final_score))

        except Exception as e:
            logging.error(f"Error calculating focus score: {str(e)}")
            return 0.0

    def _save_session(self, session: FocusSession):
        """
        Сохранение сессии в кэш

        Args:
            session: Сессия для сохранения
        """
        try:
            session_data = {
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "duration": str(session.duration) if session.duration else None,
                "focus_score": session.focus_score,
                "interruptions": session.interruptions,
                "notes": session.notes,
            }

            cache_key = f"session_{session.start_time.strftime('%Y%m%d_%H%M%S')}"
            self.cache_manager.set(cache_key, session_data)

        except Exception as e:
            logging.error(f"Error saving session: {str(e)}")
            raise
