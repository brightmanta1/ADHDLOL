"""
Модуль управления сессиями пользователей
"""

from typing import Dict, Any, Optional
import asyncio
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass
import logging


@dataclass
class UserSession:
    """Структура для хранения информации о сессии пользователя."""

    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    resources: Dict[str, Any]
    processing_queue: asyncio.Queue
    is_active: bool = True


class SessionManager:
    """Менеджер сессий пользователей."""

    def __init__(
        self, max_concurrent_sessions: int = 1000, session_timeout: int = 3600
    ):
        self.sessions: Dict[str, UserSession] = {}
        self.max_concurrent_sessions = max_concurrent_sessions
        self.session_timeout = session_timeout
        self.logger = logging.getLogger(__name__)

    async def create_session(self, user_id: str) -> Optional[str]:
        """Создание новой сессии пользователя."""
        if len(self.sessions) >= self.max_concurrent_sessions:
            self.logger.warning(
                f"Достигнут лимит активных сессий: {self.max_concurrent_sessions}"
            )
            return None

        session_id = str(uuid.uuid4())
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            resources={},
            processing_queue=asyncio.Queue(),
        )

        self.sessions[session_id] = session
        self.logger.info(
            f"Создана новая сессия: {session_id} для пользователя: {user_id}"
        )
        return session_id

    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Получение сессии по ID."""
        session = self.sessions.get(session_id)
        if session and self._is_session_valid(session):
            session.last_activity = datetime.now()
            return session
        return None

    async def end_session(self, session_id: str) -> bool:
        """Завершение сессии пользователя."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.is_active = False
            # Очистка ресурсов
            await self._cleanup_session_resources(session)
            del self.sessions[session_id]
            self.logger.info(f"Сессия завершена: {session_id}")
            return True
        return False

    async def cleanup_inactive_sessions(self):
        """Очистка неактивных сессий."""
        current_time = datetime.now()
        inactive_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if (current_time - session.last_activity).total_seconds()
            > self.session_timeout
        ]

        for session_id in inactive_sessions:
            await self.end_session(session_id)

    def _is_session_valid(self, session: UserSession) -> bool:
        """Проверка валидности сессии."""
        if not session.is_active:
            return False

        current_time = datetime.now()
        if (
            current_time - session.last_activity
        ).total_seconds() > self.session_timeout:
            session.is_active = False
            return False

        return True

    async def _cleanup_session_resources(self, session: UserSession):
        """Очистка ресурсов сессии."""
        # Очистка очереди обработки
        while not session.processing_queue.empty():
            try:
                await session.processing_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Очистка других ресурсов
        session.resources.clear()

    async def get_session_stats(self) -> Dict[str, Any]:
        """Получение статистики по сессиям."""
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": sum(1 for s in self.sessions.values() if s.is_active),
            "max_sessions": self.max_concurrent_sessions,
            "sessions_by_user": self._get_sessions_by_user(),
        }

    def _get_sessions_by_user(self) -> Dict[str, int]:
        """Получение количества сессий по пользователям."""
        user_sessions = {}
        for session in self.sessions.values():
            user_sessions[session.user_id] = user_sessions.get(session.user_id, 0) + 1
        return user_sessions
