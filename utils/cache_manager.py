"""
Модуль для управления кэшированием результатов обработки.
Поддерживает кэширование для различных типов данных и операций.
"""

import os
import json
import hashlib
import logging
import pickle
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import asdict, is_dataclass
import numpy as np


class CacheManager:
    def __init__(self, cache_dir: str = "./cache", max_age_days: int = 7):
        """
        Инициализация менеджера кэша

        Args:
            cache_dir: Директория для хранения кэша
            max_age_days: Максимальный возраст кэша в днях
        """
        self.cache_dir = cache_dir
        self.max_age = timedelta(days=max_age_days)

        # Создаем поддиректории для разных типов кэша
        self.dirs = {
            "video": os.path.join(cache_dir, "video"),
            "text": os.path.join(cache_dir, "text"),
            "concepts": os.path.join(cache_dir, "concepts"),
            "questions": os.path.join(cache_dir, "questions"),
            "models": os.path.join(cache_dir, "models"),
        }

        self._init_cache_dirs()
        logging.info(f"CacheManager initialized with cache directory: {cache_dir}")

    def _init_cache_dirs(self):
        """Инициализация директорий кэша"""
        try:
            for dir_path in self.dirs.values():
                os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            logging.error(f"Error creating cache directories: {str(e)}")
            raise

    def _generate_key(self, data: Any, prefix: str = "") -> str:
        """
        Генерация ключа кэша из данных

        Args:
            data: Данные для генерации ключа
            prefix: Префикс для ключа

        Returns:
            str: Ключ кэша
        """
        try:
            if isinstance(data, str):
                content = data.encode("utf-8")
            elif isinstance(data, dict):
                content = json.dumps(data, sort_keys=True).encode("utf-8")
            elif isinstance(data, (list, tuple)):
                content = json.dumps(list(data), sort_keys=True).encode("utf-8")
            else:
                content = str(data).encode("utf-8")

            key = hashlib.md5(content).hexdigest()
            return f"{prefix}_{key}" if prefix else key
        except Exception as e:
            logging.error(f"Error generating cache key: {str(e)}")
            return None

    def _get_cache_path(self, key: str, cache_type: str) -> str:
        """Получение пути к файлу кэша"""
        return os.path.join(self.dirs[cache_type], f"{key}.cache")

    def get(self, key: str, cache_type: str) -> Optional[Any]:
        """
        Получение данных из кэша

        Args:
            key: Ключ кэша
            cache_type: Тип кэша

        Returns:
            Any: Данные из кэша или None, если кэш не найден
        """
        try:
            cache_path = self._get_cache_path(key, cache_type)

            if not os.path.exists(cache_path):
                return None

            # Проверяем возраст кэша
            if (
                datetime.fromtimestamp(os.path.getmtime(cache_path)) + self.max_age
                < datetime.now()
            ):
                os.remove(cache_path)
                return None

            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Error reading from cache: {str(e)}")
            return None

    def set(self, key: str, data: Any, cache_type: str):
        """
        Сохранение данных в кэш

        Args:
            key: Ключ кэша
            data: Данные для сохранения
            cache_type: Тип кэша
        """
        try:
            cache_path = self._get_cache_path(key, cache_type)

            # Преобразуем dataclass в dict для сериализации
            if is_dataclass(data):
                data = asdict(data)

            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logging.error(f"Error writing to cache: {str(e)}")

    def clear(self, cache_type: Optional[str] = None):
        """
        Очистка кэша

        Args:
            cache_type: Тип кэша для очистки (если None, очищается весь кэш)
        """
        try:
            if cache_type:
                dir_path = self.dirs[cache_type]
                for file in os.listdir(dir_path):
                    os.remove(os.path.join(dir_path, file))
            else:
                for dir_path in self.dirs.values():
                    for file in os.listdir(dir_path):
                        os.remove(os.path.join(dir_path, file))
        except Exception as e:
            logging.error(f"Error clearing cache: {str(e)}")

    def get_cache_size(self, cache_type: Optional[str] = None) -> int:
        """
        Получение размера кэша в байтах

        Args:
            cache_type: Тип кэша (если None, возвращается общий размер)

        Returns:
            int: Размер кэша в байтах
        """
        try:
            total_size = 0
            if cache_type:
                dir_path = self.dirs[cache_type]
                for file in os.listdir(dir_path):
                    total_size += os.path.getsize(os.path.join(dir_path, file))
            else:
                for dir_path in self.dirs.values():
                    for file in os.listdir(dir_path):
                        total_size += os.path.getsize(os.path.join(dir_path, file))
            return total_size
        except Exception as e:
            logging.error(f"Error calculating cache size: {str(e)}")
            return 0

    def cleanup_old_cache(self):
        """Очистка устаревшего кэша"""
        try:
            current_time = datetime.now()
            for dir_path in self.dirs.values():
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if (
                        datetime.fromtimestamp(os.path.getmtime(file_path))
                        + self.max_age
                        < current_time
                    ):
                        os.remove(file_path)
        except Exception as e:
            logging.error(f"Error cleaning up old cache: {str(e)}")

    def is_cached(self, key: str, cache_type: str) -> bool:
        """
        Проверка наличия данных в кэше

        Args:
            key: Ключ кэша
            cache_type: Тип кэша

        Returns:
            bool: True если данные есть в кэше и не устарели
        """
        try:
            cache_path = self._get_cache_path(key, cache_type)

            if not os.path.exists(cache_path):
                return False

            # Проверяем возраст кэша
            return (
                datetime.fromtimestamp(os.path.getmtime(cache_path)) + self.max_age
                >= datetime.now()
            )
        except Exception as e:
            logging.error(f"Error checking cache: {str(e)}")
            return False
