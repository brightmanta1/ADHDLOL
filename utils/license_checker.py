"""
Модуль проверки лицензии и защиты кода
"""

import os
import sys
import uuid
import hashlib
import platform
from datetime import datetime
from typing import Optional
import requests


class LicenseValidator:
    def __init__(self):
        self._k = hashlib.sha256(platform.node().encode()).hexdigest()[:16]
        self._v = None
        self._t = datetime.now().timestamp()

    def _g(self, s: str) -> str:
        """Генерация хэша."""
        return hashlib.sha256(f"{s}{self._k}".encode()).hexdigest()

    def _c(self) -> bool:
        """Проверка валидности."""
        try:
            if not self._v:
                return False
            h = self._g(f"{self._t}")
            return h == self._v
        except:
            return False

    async def validate(self, key: str) -> bool:
        """Валидация лицензии."""
        try:
            # Проверка hardware ID
            hw_id = self._get_hardware_id()

            # Проверка через API
            response = await self._verify_license(key, hw_id)
            if not response:
                return False

            self._v = self._g(response)
            return self._c()
        except:
            return False

    def _get_hardware_id(self) -> str:
        """Получение уникального ID оборудования."""
        try:
            # Сбор информации об оборудовании
            system = platform.system()
            machine = platform.machine()
            processor = platform.processor()

            # Получение MAC-адреса
            from uuid import getnode

            mac = getnode()

            # Создание уникального ID
            hw_str = f"{system}:{machine}:{processor}:{mac}"
            return hashlib.sha256(hw_str.encode()).hexdigest()
        except:
            return ""

    async def _verify_license(self, key: str, hw_id: str) -> Optional[str]:
        """Проверка лицензии через API."""
        try:
            # Здесь должен быть запрос к вашему API для проверки лицензии
            # Временная заглушка для демонстрации
            if len(key) != 32:
                return None
            return f"{key}:{hw_id}"
        except:
            return None


class CodeProtector:
    """Защита кода от несанкционированного использования."""

    @staticmethod
    def verify_environment() -> bool:
        """Проверка среды выполнения."""
        try:
            # Проверка на отладку
            if sys.gettrace():
                return False

            # Проверка на виртуальную среду
            if CodeProtector._check_virtual_environment():
                return False

            # Проверка на подмену модулей
            if CodeProtector._check_module_integrity():
                return False

            return True
        except:
            return False

    @staticmethod
    def _check_virtual_environment() -> bool:
        """Проверка на виртуальную среду."""
        try:
            # Проверка Docker
            if os.path.exists("/.dockerenv"):
                return True

            # Проверка VMware
            if os.path.exists("/sys/class/dmi/id/product_name"):
                with open("/sys/class/dmi/id/product_name") as f:
                    if "VMware" in f.read():
                        return True

            return False
        except:
            return False

    @staticmethod
    def _check_module_integrity() -> bool:
        """Проверка целостности модулей."""
        try:
            # Проверка хэшей критических модулей
            critical_modules = [
                "service_manager.py",
                "personalization.py",
                "database.py",
            ]

            for module in critical_modules:
                if not CodeProtector._verify_file_hash(module):
                    return True

            return False
        except:
            return True

    @staticmethod
    def _verify_file_hash(filename: str) -> bool:
        """Проверка хэша файла."""
        try:
            # Здесь должна быть проверка хэша файла
            # Временная заглушка
            return True
        except:
            return False


# Глобальный экземпляр для проверки лицензии
license_validator = LicenseValidator()
code_protector = CodeProtector()
