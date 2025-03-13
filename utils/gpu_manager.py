"""
Модуль для управления GPU ресурсами.
"""

import logging
import torch
import psutil
import GPUtil
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class GPUStats:
    """Статистика использования GPU"""

    device_id: int
    memory_total: float
    memory_used: float
    memory_free: float
    gpu_load: float
    temperature: float


class GPUManager:
    def __init__(self):
        """Инициализация менеджера GPU"""
        self.available = torch.cuda.is_available()
        if self.available:
            self.device_count = torch.cuda.device_count()
            self.devices = {
                i: torch.device(f"cuda:{i}") for i in range(self.device_count)
            }
            logging.info(f"Initialized GPUManager with {self.device_count} GPU devices")
        else:
            self.device_count = 0
            self.devices = {}
            logging.warning("No GPU devices available")

    def get_optimal_device(
        self, required_memory: Optional[float] = None
    ) -> torch.device:
        """
        Получение оптимального устройства для выполнения задачи

        Args:
            required_memory: Требуемый объем памяти в GB

        Returns:
            torch.device: Оптимальное устройство (GPU или CPU)
        """
        if not self.available:
            return torch.device("cpu")

        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return torch.device("cpu")

            # Находим GPU с наибольшим количеством свободной памяти
            best_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)

            if required_memory and best_gpu.memoryFree < required_memory:
                logging.warning(
                    f"Not enough GPU memory. Required: {required_memory}GB, Available: {best_gpu.memoryFree}GB"
                )
                return torch.device("cpu")

            return self.devices[best_gpu.id]

        except Exception as e:
            logging.error(f"Error selecting GPU device: {str(e)}")
            return torch.device("cpu")

    def get_gpu_stats(self) -> List[GPUStats]:
        """
        Получение статистики использования GPU

        Returns:
            List[GPUStats]: Статистика по каждому GPU
        """
        if not self.available:
            return []

        try:
            stats = []
            gpus = GPUtil.getGPUs()

            for gpu in gpus:
                stats.append(
                    GPUStats(
                        device_id=gpu.id,
                        memory_total=gpu.memoryTotal,
                        memory_used=gpu.memoryUsed,
                        memory_free=gpu.memoryFree,
                        gpu_load=gpu.load,
                        temperature=gpu.temperature,
                    )
                )

            return stats
        except Exception as e:
            logging.error(f"Error getting GPU stats: {str(e)}")
            return []

    def cleanup(self):
        """Очистка памяти GPU"""
        if self.available:
            try:
                torch.cuda.empty_cache()
                logging.info("GPU memory cache cleared")
            except Exception as e:
                logging.error(f"Error cleaning up GPU memory: {str(e)}")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Получение информации об использовании памяти

        Returns:
            Dict[str, float]: Информация о памяти в GB
        """
        try:
            cpu_memory = psutil.Process().memory_info().rss / 1024**3
            gpu_memory = 0.0

            if self.available:
                gpu_memory = torch.cuda.max_memory_allocated() / 1024**3

            return {"cpu_memory_gb": cpu_memory, "gpu_memory_gb": gpu_memory}
        except Exception as e:
            logging.error(f"Error getting memory usage: {str(e)}")
            return {"cpu_memory_gb": 0.0, "gpu_memory_gb": 0.0}

    def is_gpu_suitable(
        self, required_memory: float, max_temperature: float = 80.0
    ) -> bool:
        """
        Проверка пригодности GPU для задачи

        Args:
            required_memory: Требуемый объем памяти в GB
            max_temperature: Максимально допустимая температура

        Returns:
            bool: True если GPU подходит для задачи
        """
        if not self.available:
            return False

        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return False

            best_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)

            return (
                best_gpu.memoryFree >= required_memory
                and best_gpu.temperature <= max_temperature
            )

        except Exception as e:
            logging.error(f"Error checking GPU suitability: {str(e)}")
            return False
