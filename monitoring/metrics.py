"""
Конфигурация метрик для мониторинга
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time

# Метрики запросов
REQUEST_COUNT = Counter(
    "adhd_request_total", "Total number of requests", ["type", "status"]
)

REQUEST_LATENCY = Histogram(
    "adhd_request_latency_seconds", "Request latency in seconds", ["type"]
)

# Метрики ресурсов
RESOURCE_USAGE = Gauge(
    "adhd_resource_usage", "Resource usage metrics", ["resource_type"]
)

ACTIVE_SESSIONS = Gauge("adhd_active_sessions", "Number of active sessions")

QUEUE_SIZE = Gauge("adhd_task_queue_size", "Number of tasks in queue")

# Метрики Celery
TASK_STATUS = Counter(
    "adhd_celery_task_status", "Celery task status", ["task_name", "status"]
)

TASK_LATENCY = Histogram(
    "adhd_celery_task_latency_seconds", "Celery task latency in seconds", ["task_name"]
)

# Метрики кэша
CACHE_HITS = Counter("adhd_cache_hits_total", "Total number of cache hits")

CACHE_MISSES = Counter("adhd_cache_misses_total", "Total number of cache misses")

# Системная информация
SYSTEM_INFO = Info("adhd_system", "System information")


class MetricsCollector:
    """Сборщик метрик."""

    @staticmethod
    def track_request(request_type: str):
        """Декоратор для отслеживания запросов."""

        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    REQUEST_COUNT.labels(type=request_type, status="success").inc()
                    return result
                except Exception as e:
                    REQUEST_COUNT.labels(type=request_type, status="error").inc()
                    raise e
                finally:
                    REQUEST_LATENCY.labels(type=request_type).observe(
                        time.time() - start_time
                    )

            return wrapper

        return decorator

    @staticmethod
    def track_task(task_name: str):
        """Декоратор для отслеживания задач Celery."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    TASK_STATUS.labels(task_name=task_name, status="success").inc()
                    return result
                except Exception as e:
                    TASK_STATUS.labels(task_name=task_name, status="error").inc()
                    raise e
                finally:
                    TASK_LATENCY.labels(task_name=task_name).observe(
                        time.time() - start_time
                    )

            return wrapper

        return decorator

    @staticmethod
    def update_resource_metrics(metrics: dict):
        """Обновление метрик ресурсов."""
        for resource_type, value in metrics.items():
            RESOURCE_USAGE.labels(resource_type=resource_type).set(value)

    @staticmethod
    def update_session_count(count: int):
        """Обновление количества активных сессий."""
        ACTIVE_SESSIONS.set(count)

    @staticmethod
    def update_queue_size(size: int):
        """Обновление размера очереди задач."""
        QUEUE_SIZE.set(size)

    @staticmethod
    def track_cache_operation(hit: bool):
        """Отслеживание операций с кэшем."""
        if hit:
            CACHE_HITS.inc()
        else:
            CACHE_MISSES.inc()

    @staticmethod
    def update_system_info(info: dict):
        """Обновление системной информации."""
        SYSTEM_INFO.info(info)
