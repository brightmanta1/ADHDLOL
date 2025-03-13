"""
Модуль для работы с Supabase
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from dotenv import load_dotenv
from supabase import create_client, Client
from postgrest import APIError

# Загрузка переменных окружения
load_dotenv()


class Database:
    """Класс для работы с Supabase."""

    def __init__(self):
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL", ""), os.getenv("SUPABASE_KEY", "")
        )
        self.logger = logging.getLogger(__name__)

    async def create_user_profile(
        self, user_id: str, profile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Создание профиля пользователя."""
        try:
            response = (
                await self.supabase.table("user_profiles")
                .insert(
                    {
                        "user_id": user_id,
                        "created_at": datetime.now().isoformat(),
                        **profile_data,
                    }
                )
                .execute()
            )
            return response.data[0]
        except APIError as e:
            self.logger.error(f"Ошибка при создании профиля: {str(e)}")
            raise

    async def update_user_profile(
        self, user_id: str, profile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Обновление профиля пользователя."""
        try:
            response = (
                await self.supabase.table("user_profiles")
                .update(profile_data)
                .eq("user_id", user_id)
                .execute()
            )
            return response.data[0]
        except APIError as e:
            self.logger.error(f"Ошибка при обновлении профиля: {str(e)}")
            raise

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Получение профиля пользователя."""
        try:
            response = (
                await self.supabase.table("user_profiles")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )
            return response.data[0] if response.data else None
        except APIError as e:
            self.logger.error(f"Ошибка при получении профиля: {str(e)}")
            raise

    async def save_learning_session(
        self, session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Сохранение сессии обучения."""
        try:
            response = (
                await self.supabase.table("learning_sessions")
                .insert({"created_at": datetime.now().isoformat(), **session_data})
                .execute()
            )
            return response.data[0]
        except APIError as e:
            self.logger.error(f"Ошибка при сохранении сессии: {str(e)}")
            raise

    async def save_content_metadata(
        self, content_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Сохранение метаданных контента."""
        try:
            response = (
                await self.supabase.table("content_metadata")
                .insert({"created_at": datetime.now().isoformat(), **content_data})
                .execute()
            )
            return response.data[0]
        except APIError as e:
            self.logger.error(f"Ошибка при сохранении метаданных: {str(e)}")
            raise

    async def get_user_learning_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Получение истории обучения пользователя."""
        try:
            response = (
                await self.supabase.table("learning_sessions")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )
            return response.data
        except APIError as e:
            self.logger.error(f"Ошибка при получении истории: {str(e)}")
            raise

    async def save_analytics_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Сохранение аналитического события."""
        try:
            response = (
                await self.supabase.table("analytics_events")
                .insert({"timestamp": datetime.now().isoformat(), **event_data})
                .execute()
            )
            return response.data[0]
        except APIError as e:
            self.logger.error(f"Ошибка при сохранении события: {str(e)}")
            raise

    async def get_user_analytics(
        self, user_id: str, event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Получение аналитики пользователя."""
        try:
            query = (
                self.supabase.table("analytics_events")
                .select("*")
                .eq("user_id", user_id)
            )
            if event_type:
                query = query.eq("event_type", event_type)
            response = await query.order("timestamp", desc=True).execute()
            return response.data
        except APIError as e:
            self.logger.error(f"Ошибка при получении аналитики: {str(e)}")
            raise

    async def save_resource_usage(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Сохранение данных об использовании ресурсов."""
        try:
            response = (
                await self.supabase.table("resource_usage")
                .insert({"timestamp": datetime.now().isoformat(), **usage_data})
                .execute()
            )
            return response.data[0]
        except APIError as e:
            self.logger.error(f"Ошибка при сохранении использования ресурсов: {str(e)}")
            raise
