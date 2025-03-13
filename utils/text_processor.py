"""
Модуль для обработки текстового контента.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from transformers import pipeline
import torch
from .question_generator import QuestionGenerator
from .concept_extractor import ConceptExtractor
from .cache_manager import CacheManager
from .result_printer import print_processing_results, ContentType


@dataclass
class ProcessedText:
    """Структура данных для хранения обработанного текста"""

    original_text: str
    simplified_text: str
    topics: List[str]
    terms: Dict[str, str]
    tags: List[str]
    questions: List[Dict]


class TextProcessor:
    def __init__(self, use_gpu: bool = True):
        """
        Инициализация процессора текста

        Args:
            use_gpu: Использовать ли GPU для обработки
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Инициализация моделей
        self.summarizer = pipeline(
            "summarization", model="facebook/bart-large-cnn", device=self.device
        )
        self.question_generator = QuestionGenerator(use_gpu=use_gpu)
        self.concept_extractor = ConceptExtractor()
        self.cache_manager = CacheManager(cache_dir="./cache")

        logging.info(f"TextProcessor initialized using device: {self.device}")

    def _process_text_internal(self, text: str) -> Dict[str, Any]:
        """
        Внутренний метод для обработки текста

        Args:
            text: Исходный текст для обработки

        Returns:
            Dict[str, Any]: Результаты обработки текста
        """
        try:
            # Очистка текста
            cleaned_text = self._clean_text(text)

            # Обработка текста
            simplified = self.simplify_text(cleaned_text)
            topics = self._extract_topics(cleaned_text)
            terms = self._extract_terms(cleaned_text)
            tags = self._generate_tags(cleaned_text)
            questions = self.question_generator.generate_questions(cleaned_text)

            return {
                "status": "success",
                "original_text": text,
                "simplified_text": simplified,
                "topics": topics,
                "terms": terms,
                "tags": tags,
                "questions": questions,
                "text_info": {
                    "length": len(text),
                    "simplified_length": len(simplified),
                    "topic_count": len(topics),
                    "term_count": len(terms),
                    "question_count": len(questions),
                },
            }
        except Exception as e:
            logging.error(f"Error processing text: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "text_info": {"length": len(text)},
            }

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Обработка текста и возврат результатов.

        Args:
            text: Текст для обработки

        Returns:
            Dict[str, Any]: Результаты обработки текста
        """
        try:
            # Check cache first
            cache_key = self.cache_manager.generate_key(text, prefix="text")
            cached_result = self.cache_manager.get(cache_key, "text")
            if cached_result:
                print_processing_results(cached_result, ContentType.TEXT)
                return cached_result

            # Process text if not in cache
            result = self._process_text_internal(text)

            # Cache results
            if result["status"] == "success":
                self.cache_manager.set(cache_key, result, "text")

            # Print results
            print_processing_results(result, ContentType.TEXT)

            return result

        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "text_info": {"length": len(text)},
            }
            print_processing_results(error_result, ContentType.TEXT)
            return error_result

    def simplify_text(self, text: str) -> str:
        """
        Упрощение текста для лучшего понимания

        Args:
            text: Текст для упрощения

        Returns:
            str: Упрощенный текст
        """
        try:
            cache_key = self.cache_manager.generate_key(text, prefix="simplified")
            cached_result = self.cache_manager.get(cache_key, "text")
            if cached_result:
                return cached_result

            summary = self.summarizer(
                text, max_length=130, min_length=30, do_sample=False
            )
            result = summary[0]["summary_text"]

            self.cache_manager.set(cache_key, result, "text")
            return result
        except Exception as e:
            logging.error(f"Error simplifying text: {str(e)}")
            return text

    def _clean_text(self, text: str) -> str:
        """
        Очистка и нормализация текста

        Args:
            text: Текст для очистки

        Returns:
            str: Очищенный текст
        """
        try:
            # Удаление лишних пробелов и специальных символов
            cleaned = " ".join(text.split())
            return cleaned
        except Exception as e:
            logging.error(f"Error cleaning text: {str(e)}")
            return text

    def _extract_topics(self, text: str) -> List[str]:
        """
        Извлечение основных тем из текста

        Args:
            text: Текст для анализа

        Returns:
            List[str]: Список основных тем
        """
        try:
            cache_key = self.cache_manager.generate_key(text, prefix="topics")
            cached_result = self.cache_manager.get(cache_key, "text")
            if cached_result:
                return cached_result

            summary = self.summarizer(
                text, max_length=50, min_length=10, do_sample=False
            )
            topics = summary[0]["summary_text"].split(".")
            result = [t.strip() for t in topics if t.strip()][:5]

            self.cache_manager.set(cache_key, result, "text")
            return result
        except Exception as e:
            logging.error(f"Error extracting topics: {str(e)}")
            return []

    def _extract_terms(self, text: str) -> Dict[str, str]:
        """
        Извлечение ключевых терминов и их контекста

        Args:
            text: Текст для анализа

        Returns:
            Dict[str, str]: Словарь терминов и их определений
        """
        try:
            cache_key = self.cache_manager.generate_key(text, prefix="terms")
            cached_result = self.cache_manager.get(cache_key, "text")
            if cached_result:
                return cached_result

            result = self.concept_extractor.extract_terms(text)
            self.cache_manager.set(cache_key, result, "text")
            return result
        except Exception as e:
            logging.error(f"Error extracting terms: {str(e)}")
            return {}

    def _generate_tags(self, text: str) -> List[str]:
        """
        Генерация тегов на основе текста

        Args:
            text: Текст для анализа

        Returns:
            List[str]: Список сгенерированных тегов
        """
        try:
            cache_key = self.cache_manager.generate_key(text, prefix="tags")
            cached_result = self.cache_manager.get(cache_key, "text")
            if cached_result:
                return cached_result

            result = self.concept_extractor.generate_tags(text)
            self.cache_manager.set(cache_key, result, "text")
            return result
        except Exception as e:
            logging.error(f"Error generating tags: {str(e)}")
            return []
