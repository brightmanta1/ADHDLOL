"""
Конфигурация Celery и Redis
"""

import os
from celery import Celery
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Создание Celery приложения
celery_app = Celery(
    "ADHDLearningCompanion",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)

# Настройка Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 час
    worker_max_tasks_per_child=100,
    worker_prefetch_multiplier=1,
)

# Настройка Redis для кэширования
from redis import Redis

redis_client = Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True
)


# Задачи Celery
@celery_app.task(bind=True, name="process_content")
def process_content(self, content_type: str, content_data: dict) -> dict:
    """Обработка контента в фоновом режиме."""
    from .service_manager import ServiceManager

    service = ServiceManager()
    return service._process_request({"type": content_type, "content": content_data})


@celery_app.task(bind=True, name="create_schedule")
def create_schedule(self, user_id: str) -> dict:
    """Создание расписания в фоновом режиме."""
    from .service_manager import ServiceManager

    service = ServiceManager()
    return service.scheduler.create_personalized_schedule(user_id)


@celery_app.task(bind=True, name="cleanup_old_sessions")
def cleanup_old_sessions(self) -> None:
    """Очистка старых сессий."""
    from .service_manager import ServiceManager

    service = ServiceManager()
    service.session_manager.cleanup_inactive_sessions()


# Периодические задачи
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Настройка периодических задач."""
    # Очистка сессий каждый час
    sender.add_periodic_task(3600.0, cleanup_old_sessions.s(), name="cleanup-sessions")
