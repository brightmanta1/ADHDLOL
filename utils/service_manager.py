"""
Модуль управления сервисом для обработки запросов пользователей
"""

from typing import Dict, Any, Optional
import asyncio
import uuid
from datetime import datetime
import logging
import os
import pinecone
from dotenv import load_dotenv
from .session_manager import SessionManager
from .resource_manager import ResourceManager, Task, TaskPriority, ResourceType
from .learning_agent import ContentProcessor
from .text_processor import TextProcessor
from .database import Database
from .personalization import PersonalizationEngine
from .scheduler import LearningScheduler
from .license_checker import license_validator, code_protector

# Загрузка переменных окружения
load_dotenv()


class ServiceManager:
    """Менеджер сервиса для обработки запросов пользователей."""

    def __init__(self):
        # Проверка среды выполнения
        if not code_protector.verify_environment():
            raise RuntimeError("Недопустимая среда выполнения")

        # Проверка лицензии
        self._license_key = os.getenv("LICENSE_KEY")
        if not self._license_key:
            raise RuntimeError("Отсутствует ключ лицензии")

        # Инициализация Pinecone
        self._init_pinecone()

        # Инициализация компонентов
        self.session_manager = SessionManager(
            max_concurrent_sessions=int(os.getenv("MAX_CONCURRENT_SESSIONS", 1000)),
            session_timeout=int(os.getenv("SESSION_TIMEOUT", 3600)),
        )
        self.resource_manager = ResourceManager()
        self.content_processor = ContentProcessor()
        self.text_processor = TextProcessor()
        self.database = Database()
        self.personalization = PersonalizationEngine(self.pinecone_index)
        self.scheduler = LearningScheduler(self.personalization)
        self.logger = logging.getLogger(__name__)

        # Запуск фоновых задач
        self.background_tasks = set()
        self._start_background_tasks()

    def _init_pinecone(self):
        """Инициализация Pinecone."""
        try:
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp"),
            )
            index_name = os.getenv("PINECONE_INDEX_NAME", "learning-content")

            # Создание индекса, если он не существует
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=384,  # Размерность для all-MiniLM-L6-v2
                    metric="cosine",
                    pods=1,
                    replicas=1,
                    pod_type="p1.x1",
                )

            self.pinecone_index = pinecone.Index(index_name)
            self.logger.info("Pinecone успешно инициализирован")

        except Exception as e:
            self.logger.error(f"Ошибка при инициализации Pinecone: {str(e)}")
            self.pinecone_index = None

    def _start_background_tasks(self):
        """Запуск фоновых задач."""
        loop = asyncio.get_event_loop()

        # Задача обработки очереди
        task = loop.create_task(self.resource_manager.process_tasks())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

        # Задача очистки сессий
        task = loop.create_task(self._cleanup_sessions())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

        # Задача сохранения статистики
        task = loop.create_task(self._save_stats())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def _cleanup_sessions(self):
        """Периодическая очистка неактивных сессий."""
        while True:
            await self.session_manager.cleanup_inactive_sessions()
            await asyncio.sleep(60)

    async def _save_stats(self):
        """Периодическое сохранение статистики."""
        while True:
            try:
                stats = await self.get_service_stats()
                await self.database.save_resource_usage(
                    {"stats": stats, "timestamp": datetime.now().isoformat()}
                )
            except Exception as e:
                self.logger.error(f"Ошибка при сохранении статистики: {str(e)}")
            await asyncio.sleep(300)  # каждые 5 минут

    async def handle_request(
        self, user_id: str, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Обработка запроса пользователя."""
        try:
            # Проверка лицензии
            if not await license_validator.validate(self._license_key):
                raise RuntimeError("Недействительная лицензия")

            # Получение или создание сессии
            session_id = request_data.get("session_id")
            if not session_id:
                session_id = await self.session_manager.create_session(user_id)
                if not session_id:
                    return {"status": "error", "message": "Не удалось создать сессию"}

            # Проверка сессии
            session = await self.session_manager.get_session(session_id)
            if not session:
                return {"status": "error", "message": "Недействительная сессия"}

            # Получение профиля пользователя
            user_profile = await self.database.get_user_profile(user_id)
            if not user_profile:
                # Создание нового профиля
                user_profile = await self.database.create_user_profile(
                    user_id,
                    {
                        "preferences": {},
                        "learning_history": [],
                        "created_at": datetime.now().isoformat(),
                    },
                )

            # Персонализация запроса
            request_data = await self._personalize_request(user_id, request_data)

            # Создание задачи
            task = self._create_task(user_id, session_id, request_data)

            # Добавление задачи в очередь
            status = await self.resource_manager.submit_task(task)

            # Сохранение события
            await self.database.save_analytics_event(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "event_type": "request",
                    "data": request_data,
                    "status": status,
                }
            )

            return {
                "status": status,
                "task_id": task.task_id,
                "session_id": session_id,
                "message": (
                    "Задача добавлена в очередь"
                    if status == "queued"
                    else "Ошибка при добавлении задачи"
                ),
            }

        except Exception as e:
            self.logger.error(f"Ошибка при обработке запроса: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _personalize_request(
        self, user_id: str, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Персонализация запроса на основе профиля пользователя."""
        try:
            # Получение оптимального времени обучения
            optimal_time = await self.scheduler.create_personalized_schedule(user_id)

            # Обновление параметров запроса
            request_data.update(
                {
                    "optimal_time": optimal_time,
                    "personalization": {
                        "user_id": user_id,
                        "timestamp": datetime.now().isoformat(),
                    },
                }
            )

            return request_data
        except Exception as e:
            self.logger.error(f"Ошибка при персонализации запроса: {str(e)}")
            return request_data

    def _create_task(
        self, user_id: str, session_id: str, request_data: Dict[str, Any]
    ) -> Task:
        """Создание задачи из запроса."""
        task_id = str(uuid.uuid4())

        # Определение приоритета задачи
        priority = self._determine_priority(request_data)

        # Определение требуемых ресурсов
        resource_requirements = self._calculate_resource_requirements(request_data)

        # Создание корутины для выполнения задачи
        coroutine = self._process_request(request_data)

        return Task(
            task_id=task_id,
            user_id=user_id,
            session_id=session_id,
            priority=priority,
            resource_requirements=resource_requirements,
            created_at=datetime.now(),
            timeout=float(request_data.get("timeout", 300)),
            coroutine=coroutine,
        )

    def _determine_priority(self, request_data: Dict[str, Any]) -> TaskPriority:
        """Определение приоритета задачи."""
        priority_map = {
            "low": TaskPriority.LOW,
            "medium": TaskPriority.MEDIUM,
            "high": TaskPriority.HIGH,
            "critical": TaskPriority.CRITICAL,
        }
        return priority_map.get(
            request_data.get("priority", "medium"), TaskPriority.MEDIUM
        )

    def _calculate_resource_requirements(
        self, request_data: Dict[str, Any]
    ) -> Dict[ResourceType, float]:
        """Расчет требуемых ресурсов для задачи."""
        # Базовые требования
        requirements = {
            ResourceType.CPU: 10.0,  # 10% CPU
            ResourceType.MEMORY: 512.0,  # 512MB RAM
            ResourceType.API_CALL: 1.0,  # 1 API вызов
        }

        # Дополнительные требования в зависимости от типа задачи
        task_type = request_data.get("type", "")
        if task_type == "video_processing":
            requirements.update(
                {
                    ResourceType.CPU: 30.0,
                    ResourceType.GPU: 50.0,
                    ResourceType.MEMORY: 1024.0,
                }
            )
        elif task_type == "text_processing":
            requirements.update({ResourceType.CPU: 20.0, ResourceType.MEMORY: 768.0})

        return requirements

    async def _process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запроса с использованием соответствующих процессоров."""
        try:
            task_type = request_data.get("type", "")
            content = request_data.get("content", {})

            if task_type == "video_processing":
                # Обработка видео через ContentProcessor
                result = await self.content_processor.process_content(
                    content=content.get("url", ""),
                    content_type=content.get("format", "youtube"),
                    title=content.get("title", ""),
                )

                # Сохранение метаданных
                await self.database.save_content_metadata(
                    {"type": "video", "url": content.get("url", ""), "metadata": result}
                )

                return {"status": "success", "type": "video", "result": result}

            elif task_type == "text_processing":
                # Обработка текста через TextProcessor
                result = self.text_processor.process_text(content.get("text", ""))

                # Сохранение метаданных
                await self.database.save_content_metadata(
                    {
                        "type": "text",
                        "content": content.get("text", ""),
                        "metadata": {"topics": result.topics, "tags": result.tags},
                    }
                )

                return {
                    "status": "success",
                    "type": "text",
                    "result": {
                        "simplified": result.simplified,
                        "topics": result.topics,
                        "highlighted_terms": result.highlighted_terms,
                        "tags": result.tags,
                    },
                }

            else:
                raise ValueError(f"Неизвестный тип задачи: {task_type}")

        except Exception as e:
            self.logger.error(f"Ошибка при обработке запроса: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def get_service_stats(self) -> Dict[str, Any]:
        """Получение статистики сервиса."""
        session_stats = await self.session_manager.get_session_stats()
        resource_stats = await self.resource_manager.get_resource_stats()

        return {
            "sessions": session_stats,
            "resources": resource_stats,
            "background_tasks": len(self.background_tasks),
        }
