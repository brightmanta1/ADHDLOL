"""
Модуль управления ресурсами и очередями задач
"""

from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum


class ResourceType(Enum):
    """Типы ресурсов системы."""

    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    API_CALL = "api_call"


class TaskPriority(Enum):
    """Приоритеты задач."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ResourceLimit:
    """Лимиты ресурсов."""

    max_value: float
    current_value: float = 0.0
    reserved_value: float = 0.0


@dataclass
class Task:
    """Структура задачи."""

    task_id: str
    user_id: str
    session_id: str
    priority: TaskPriority
    resource_requirements: Dict[ResourceType, float]
    created_at: datetime
    timeout: float
    coroutine: Any
    status: str = "pending"


class ResourceManager:
    """Менеджер ресурсов и очередей задач."""

    def __init__(self):
        self.resource_limits: Dict[ResourceType, ResourceLimit] = {
            ResourceType.CPU: ResourceLimit(max_value=100.0),  # 100% CPU
            ResourceType.GPU: ResourceLimit(max_value=100.0),  # 100% GPU
            ResourceType.MEMORY: ResourceLimit(max_value=1024 * 8),  # 8GB RAM
            ResourceType.STORAGE: ResourceLimit(max_value=1024 * 100),  # 100GB Storage
            ResourceType.API_CALL: ResourceLimit(max_value=100.0),  # 100 calls/sec
        }

        self.tasks: Dict[str, Task] = {}
        self.task_queues: Dict[TaskPriority, asyncio.PriorityQueue] = {
            priority: asyncio.PriorityQueue() for priority in TaskPriority
        }

        self.logger = logging.getLogger(__name__)
        self.processing_lock = asyncio.Lock()

    async def submit_task(self, task: Task) -> str:
        """Добавление задачи в очередь."""
        try:
            # Проверка доступности ресурсов
            if not self._can_allocate_resources(task.resource_requirements):
                self.logger.warning(f"Недостаточно ресурсов для задачи {task.task_id}")
                return "rejected"

            # Добавление в очередь
            await self.task_queues[task.priority].put(
                (datetime.now().timestamp(), task)
            )
            self.tasks[task.task_id] = task

            self.logger.info(
                f"Задача {task.task_id} добавлена в очередь с приоритетом {task.priority}"
            )
            return "queued"

        except Exception as e:
            self.logger.error(f"Ошибка при добавлении задачи: {str(e)}")
            return "error"

    async def process_tasks(self):
        """Обработка задач из очереди."""
        while True:
            try:
                async with self.processing_lock:
                    # Обработка задач по приоритетам
                    for priority in reversed(TaskPriority):
                        queue = self.task_queues[priority]
                        while not queue.empty():
                            _, task = await queue.get()

                            if self._can_allocate_resources(task.resource_requirements):
                                await self._process_task(task)
                            else:
                                # Возвращаем задачу в очередь
                                await queue.put((datetime.now().timestamp(), task))
                                break

                await asyncio.sleep(0.1)  # Предотвращение перегрузки CPU

            except Exception as e:
                self.logger.error(f"Ошибка при обработке задач: {str(e)}")
                await asyncio.sleep(1)  # Пауза при ошибке

    async def _process_task(self, task: Task):
        """Обработка отдельной задачи."""
        try:
            # Выделение ресурсов
            self._allocate_resources(task.resource_requirements)
            task.status = "processing"

            # Выполнение задачи
            result = await asyncio.wait_for(task.coroutine, timeout=task.timeout)

            task.status = "completed"
            return result

        except asyncio.TimeoutError:
            task.status = "timeout"
            self.logger.warning(f"Таймаут задачи {task.task_id}")

        except Exception as e:
            task.status = "error"
            self.logger.error(f"Ошибка при выполнении задачи {task.task_id}: {str(e)}")

        finally:
            # Освобождение ресурсов
            self._release_resources(task.resource_requirements)

    def _can_allocate_resources(self, requirements: Dict[ResourceType, float]) -> bool:
        """Проверка возможности выделения ресурсов."""
        for resource_type, required_value in requirements.items():
            limit = self.resource_limits[resource_type]
            if limit.current_value + required_value > limit.max_value:
                return False
        return True

    def _allocate_resources(self, requirements: Dict[ResourceType, float]):
        """Выделение ресурсов."""
        for resource_type, required_value in requirements.items():
            self.resource_limits[resource_type].current_value += required_value

    def _release_resources(self, requirements: Dict[ResourceType, float]):
        """Освобождение ресурсов."""
        for resource_type, required_value in requirements.items():
            self.resource_limits[resource_type].current_value -= required_value

    async def get_resource_stats(self) -> Dict[str, Any]:
        """Получение статистики по ресурсам."""
        return {
            "resources": {
                r_type.value: {
                    "max": limit.max_value,
                    "current": limit.current_value,
                    "available": limit.max_value - limit.current_value,
                }
                for r_type, limit in self.resource_limits.items()
            },
            "tasks": {
                "total": len(self.tasks),
                "by_status": self._get_tasks_by_status(),
                "by_priority": self._get_tasks_by_priority(),
            },
        }

    def _get_tasks_by_status(self) -> Dict[str, int]:
        """Получение количества задач по статусам."""
        status_counts = {}
        for task in self.tasks.values():
            status_counts[task.status] = status_counts.get(task.status, 0) + 1
        return status_counts

    def _get_tasks_by_priority(self) -> Dict[str, int]:
        """Получение количества задач по приоритетам."""
        priority_counts = {}
        for task in self.tasks.values():
            priority_counts[task.priority.name] = (
                priority_counts.get(task.priority.name, 0) + 1
            )
        return priority_counts
