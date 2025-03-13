"""
Модуль обфускации кода
"""

import base64
import zlib
import random
import string
from typing import Any, Callable


class CodeObfuscator:
    """Класс для обфускации кода."""

    @staticmethod
    def obfuscate_string(s: str) -> str:
        """Обфускация строки."""
        try:
            # Сжатие и кодирование
            compressed = zlib.compress(s.encode())
            encoded = base64.b85encode(compressed)

            # Добавление "шума"
            noise = "".join(random.choices(string.ascii_letters, k=8))
            return f"{noise}{encoded.decode()}"
        except:
            return s

    @staticmethod
    def deobfuscate_string(s: str) -> str:
        """Деобфускация строки."""
        try:
            # Удаление "шума"
            encoded = s[8:]

            # Декодирование и распаковка
            decoded = base64.b85decode(encoded.encode())
            decompressed = zlib.decompress(decoded)
            return decompressed.decode()
        except:
            return s

    @staticmethod
    def obfuscate_function(func: Callable) -> Callable:
        """Декоратор для обфускации функции."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                # Получение исходного кода функции
                source = func.__code__

                # Обфускация имен переменных и аргументов
                co_varnames = tuple(
                    CodeObfuscator.obfuscate_string(name) for name in source.co_varnames
                )

                # Создание нового объекта кода
                new_code = type(source)(
                    source.co_argcount,
                    source.co_posonlyargcount,
                    source.co_kwonlyargcount,
                    source.co_nlocals,
                    source.co_stacksize,
                    source.co_flags,
                    source.co_code,
                    source.co_consts,
                    source.co_names,
                    co_varnames,
                    source.co_filename,
                    source.co_name,
                    source.co_firstlineno,
                    source.co_lnotab,
                    source.co_freevars,
                    source.co_cellvars,
                )

                # Создание новой функции
                new_func = type(func)(
                    new_code,
                    func.__globals__,
                    func.__name__,
                    func.__defaults__,
                    func.__closure__,
                )

                return new_func(*args, **kwargs)
            except:
                return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def protect_class(cls: type) -> type:
        """Декоратор для защиты класса."""
        try:
            # Обфускация имени класса
            cls.__name__ = CodeObfuscator.obfuscate_string(cls.__name__)

            # Обфускация методов
            for name, method in cls.__dict__.items():
                if callable(method):
                    setattr(cls, name, CodeObfuscator.obfuscate_function(method))

            return cls
        except:
            return cls


# Пример использования:
# @CodeObfuscator.protect_class
# class ProtectedClass:
#     @CodeObfuscator.obfuscate_function
#     def protected_method(self):
#         pass
