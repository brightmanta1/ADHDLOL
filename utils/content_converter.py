"""
Модуль для конвертации различных типов контента.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import json
import yaml
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import torch
from transformers import AutoTokenizer, AutoModel
from .cache_manager import CacheManager
from .gpu_manager import GPUManager


class ContentConverter:
    def __init__(self, use_gpu: bool = True, cache_dir: str = "./cache/converter"):
        """
        Инициализация конвертера контента

        Args:
            use_gpu: Использовать ли GPU для обработки
            cache_dir: Директория для кэширования результатов
        """
        self.gpu_manager = GPUManager()
        self.device = self.gpu_manager.get_device() if use_gpu else "cpu"
        self.cache_manager = CacheManager(cache_dir=cache_dir)

        # Инициализация моделей
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-multilingual-cased"
            )
            self.model = AutoModel.from_pretrained("bert-base-multilingual-cased").to(
                self.device
            )
            logging.info(f"ContentConverter initialized using device: {self.device}")
        except Exception as e:
            logging.error(f"Error initializing models: {str(e)}")
            self.tokenizer = None
            self.model = None
            raise

    def convert(
        self, content: Any, source_type: str, target_type: str
    ) -> Union[str, Dict, List, bytes, np.ndarray]:
        """
        Конвертация контента из одного типа в другой

        Args:
            content: Контент для конвертации
            source_type: Исходный тип контента
            target_type: Целевой тип контента

        Returns:
            Union[str, Dict, List, bytes, np.ndarray]: Сконвертированный контент

        Raises:
            ValueError: Если указан неподдерживаемый тип конвертации
            Exception: При ошибке конвертации
        """
        try:
            # Проверка входных данных
            if content is None:
                raise ValueError("Content cannot be None")

            # Генерация ключа кэша
            cache_key = self.cache_manager.generate_key(
                f"{str(content)[:100]}_{source_type}_{target_type}", prefix="conversion"
            )

            # Проверка кэша
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logging.info(
                    f"Using cached result for {source_type} to {target_type} conversion"
                )
                return cached_result

            # Определение метода конвертации
            conversion_method = self._get_conversion_method(source_type, target_type)
            if not conversion_method:
                raise ValueError(
                    f"Unsupported conversion: {source_type} to {target_type}"
                )

            # Конвертация
            result = conversion_method(content)

            # Кэширование результата
            self.cache_manager.set(cache_key, result)
            logging.info(f"Successfully converted {source_type} to {target_type}")

            return result

        except ValueError as e:
            logging.error(f"Validation error in conversion: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error converting content: {str(e)}")
            raise

    def _get_conversion_method(
        self, source_type: str, target_type: str
    ) -> Optional[Callable]:
        """
        Получение метода конвертации на основе типов

        Args:
            source_type: Исходный тип
            target_type: Целевой тип

        Returns:
            Optional[Callable]: Метод конвертации или None, если конвертация не поддерживается
        """
        conversion_map = {
            ("text", "vector"): self._text_to_vector,
            ("image", "text"): self._image_to_text,
            ("json", "yaml"): self._json_to_yaml,
            ("yaml", "json"): self._yaml_to_json,
            ("csv", "json"): self._csv_to_json,
            ("json", "csv"): self._json_to_csv,
            ("text", "summary"): self._text_to_summary,
            ("image", "vector"): self._image_to_vector,
        }
        return conversion_map.get((source_type.lower(), target_type.lower()))

    def _text_to_vector(self, text: str) -> np.ndarray:
        """
        Конвертация текста в векторное представление

        Args:
            text: Текст для конвертации

        Returns:
            np.ndarray: Векторное представление текста

        Raises:
            ValueError: Если модели не инициализированы
            Exception: При ошибке конвертации
        """
        try:
            if self.tokenizer is None or self.model is None:
                raise ValueError("Models are not initialized")

            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            return embeddings.cpu().numpy()
        except ValueError as e:
            logging.error(f"Validation error in text_to_vector: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error converting text to vector: {str(e)}")
            raise

    def _image_to_text(self, image_path: Union[str, Path]) -> str:
        """
        Конвертация изображения в текстовое описание

        Args:
            image_path: Путь к изображению

        Returns:
            str: Текстовое описание изображения

        Raises:
            FileNotFoundError: Если файл не найден
            Exception: При ошибке конвертации
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Здесь должна быть реализация OCR или image captioning
            # В реальном приложении используйте модель для генерации описания
            logging.warning("Using placeholder for image_to_text conversion")
            return f"Image description for {image_path.name}"
        except FileNotFoundError as e:
            logging.error(f"File not found: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error converting image to text: {str(e)}")
            raise

    def _json_to_yaml(self, json_data: Union[str, Dict]) -> str:
        """
        Конвертация JSON в YAML

        Args:
            json_data: JSON данные (строка или словарь)

        Returns:
            str: YAML представление данных

        Raises:
            ValueError: Если данные имеют неверный формат
            Exception: При ошибке конвертации
        """
        try:
            if isinstance(json_data, str):
                json_data = json.loads(json_data)
            elif not isinstance(json_data, dict) and not isinstance(json_data, list):
                raise ValueError(f"Invalid JSON data type: {type(json_data)}")

            return yaml.dump(json_data, allow_unicode=True, sort_keys=False)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format: {str(e)}")
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            logging.error(f"Error converting JSON to YAML: {str(e)}")
            raise

    def _yaml_to_json(self, yaml_data: str) -> Dict:
        """
        Конвертация YAML в JSON

        Args:
            yaml_data: YAML данные

        Returns:
            Dict: JSON представление данных

        Raises:
            ValueError: Если данные имеют неверный формат
            Exception: При ошибке конвертации
        """
        try:
            if not isinstance(yaml_data, str):
                raise ValueError(f"YAML data must be a string, got {type(yaml_data)}")

            return yaml.safe_load(yaml_data)
        except yaml.YAMLError as e:
            logging.error(f"Invalid YAML format: {str(e)}")
            raise ValueError(f"Invalid YAML format: {str(e)}")
        except Exception as e:
            logging.error(f"Error converting YAML to JSON: {str(e)}")
            raise

    def _csv_to_json(self, csv_data: str) -> List[Dict]:
        """
        Конвертация CSV в JSON

        Args:
            csv_data: CSV данные

        Returns:
            List[Dict]: JSON представление данных

        Raises:
            ValueError: Если данные имеют неверный формат
            Exception: При ошибке конвертации
        """
        try:
            if not isinstance(csv_data, str):
                raise ValueError(f"CSV data must be a string, got {type(csv_data)}")

            df = pd.read_csv(csv_data)
            return json.loads(df.to_json(orient="records"))
        except pd.errors.ParserError as e:
            logging.error(f"Invalid CSV format: {str(e)}")
            raise ValueError(f"Invalid CSV format: {str(e)}")
        except Exception as e:
            logging.error(f"Error converting CSV to JSON: {str(e)}")
            raise

    def _json_to_csv(self, json_data: Union[str, List[Dict]]) -> str:
        """
        Конвертация JSON в CSV

        Args:
            json_data: JSON данные (строка или список словарей)

        Returns:
            str: CSV представление данных

        Raises:
            ValueError: Если данные имеют неверный формат
            Exception: При ошибке конвертации
        """
        try:
            if isinstance(json_data, str):
                json_data = json.loads(json_data)
            elif not isinstance(json_data, list):
                raise ValueError(
                    f"JSON data must be a list of dictionaries or a JSON string, got {type(json_data)}"
                )

            df = pd.DataFrame(json_data)
            return df.to_csv(index=False)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format: {str(e)}")
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            logging.error(f"Error converting JSON to CSV: {str(e)}")
            raise

    def _text_to_summary(self, text: str) -> str:
        """
        Конвертация текста в краткое содержание

        Args:
            text: Текст для суммаризации

        Returns:
            str: Краткое содержание текста

        Raises:
            ValueError: Если текст пустой
            Exception: При ошибке конвертации
        """
        try:
            if not text or not isinstance(text, str):
                raise ValueError("Text must be a non-empty string")

            # Здесь должна быть реализация суммаризации
            # В реальном приложении используйте модель для суммаризации
            logging.warning("Using placeholder for text_to_summary conversion")

            # Простая экстрактивная суммаризация
            sentences = text.split(". ")
            if len(sentences) <= 3:
                return text

            return ". ".join(sentences[:3]) + "."
        except ValueError as e:
            logging.error(f"Validation error in text_to_summary: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error converting text to summary: {str(e)}")
            raise

    def _image_to_vector(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Конвертация изображения в векторное представление

        Args:
            image_path: Путь к изображению

        Returns:
            np.ndarray: Векторное представление изображения

        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если изображение не удалось загрузить
            Exception: При ошибке конвертации
        """
        try:
            image_path = str(image_path)
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Ресайз изображения до фиксированного размера
            image = cv2.resize(image, (224, 224))

            # Нормализация
            image = image.astype(np.float32) / 255.0

            # Преобразование в тензор
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            # Здесь должна быть модель для извлечения признаков
            # В реальном приложении используйте предобученную модель
            logging.warning("Using placeholder for image_to_vector conversion")

            # Возвращаем случайный вектор для демонстрации
            return np.random.randn(1, 512).astype(np.float32)
        except FileNotFoundError as e:
            logging.error(f"File not found: {str(e)}")
            raise
        except ValueError as e:
            logging.error(f"Validation error in image_to_vector: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error converting image to vector: {str(e)}")
            raise

    def cleanup(self):
        """
        Очистка ресурсов
        """
        try:
            if self.model is not None:
                self.model = self.model.cpu()
            torch.cuda.empty_cache()
            logging.info("ContentConverter resources cleaned up")
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
