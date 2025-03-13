"""
Модуль для обработки текста с использованием различных моделей.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline,
)
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from ..utils.gpu_manager import GPUManager
from ..utils.cache_manager import CacheManager

# Загрузка необходимых ресурсов NLTK
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("averaged_perceptron_tagger")


class TextProcessor:
    def __init__(self, use_gpu: bool = True, model_path: Optional[str] = None):
        """
        Инициализация процессора текста

        Args:
            use_gpu: Использовать ли GPU для обработки
            model_path: Путь к предварительно обученным моделям
        """
        self.gpu_manager = GPUManager()
        self.device = self.gpu_manager.get_device() if use_gpu else "cpu"
        self.cache_manager = CacheManager(cache_dir="./cache/text_models")
        self.model_path = model_path or "./models"

        try:
            # Инициализация моделей
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=self.device if use_gpu else -1,
                cache_dir=self.model_path,
            )

            self.paraphraser = AutoModelForSeq2SeqLM.from_pretrained(
                "tuner007/pegasus_paraphrase", cache_dir=self.model_path
            ).to(self.device)
            self.paraphraser_tokenizer = AutoTokenizer.from_pretrained(
                "tuner007/pegasus_paraphrase", cache_dir=self.model_path
            )

            self.classifier = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=3, cache_dir=self.model_path
            ).to(self.device)
            self.classifier_tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased", cache_dir=self.model_path
            )

            self.stop_words = set(stopwords.words("english"))

            logging.info(f"TextProcessor initialized using device: {self.device}")
        except Exception as e:
            logging.error(f"Error initializing TextProcessor: {str(e)}")
            raise

    def process_text(
        self,
        text: str,
        operations: Optional[List[str]] = None,
        complexity: str = "medium",
    ) -> Dict[str, Any]:
        """
        Комплексная обработка текста

        Args:
            text: Текст для обработки
            operations: Список операций для выполнения
            complexity: Уровень сложности текста

        Returns:
            Dict[str, Any]: Результаты обработки
        """
        try:
            if not text:
                return {"error": "Empty text provided"}

            if operations is None:
                operations = ["summarize", "paraphrase", "analyze"]

            cache_key = self.cache_manager.generate_key(
                f"{text}_{'-'.join(operations)}_{complexity}", prefix="process"
            )
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result

            results = {"original_text": text, "complexity": complexity}

            if "summarize" in operations:
                results["summary"] = self.summarize_text(text)

            if "paraphrase" in operations:
                results["paraphrased"] = self.paraphrase_text(text)

            if "analyze" in operations:
                results["analysis"] = self.analyze_text(text)

            # Добавляем статистику
            results["statistics"] = self._calculate_statistics(text)

            # Кэширование результатов
            self.cache_manager.set(cache_key, results)

            return results

        except Exception as e:
            logging.error(f"Error processing text: {str(e)}")
            return {"error": str(e)}

    def _calculate_statistics(self, text: str) -> Dict[str, Any]:
        """Расчет статистики текста"""
        try:
            words = word_tokenize(text)
            sentences = sent_tokenize(text)

            return {
                "word_count": len(words),
                "sentence_count": len(sentences),
                "avg_word_length": (
                    sum(len(w) for w in words) / len(words) if words else 0
                ),
                "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
                "unique_words": len(set(w.lower() for w in words)),
            }
        except Exception as e:
            logging.error(f"Error calculating statistics: {str(e)}")
            return {}

    def summarize_text(
        self, text: str, max_length: int = 130, min_length: int = 30
    ) -> str:
        """
        Создание краткого содержания текста

        Args:
            text: Текст для обработки
            max_length: Максимальная длина summary
            min_length: Минимальная длина summary

        Returns:
            str: Краткое содержание
        """
        try:
            if not text:
                return ""

            cache_key = self.cache_manager.generate_key(
                f"{text}_{max_length}_{min_length}", prefix="summary"
            )
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result

            # Ограничиваем длину входного текста для предотвращения ошибок
            max_input_length = 1024
            if len(text) > max_input_length:
                text = text[:max_input_length]

            summary = self.summarizer(
                text, max_length=max_length, min_length=min_length, do_sample=False
            )[0]["summary_text"]

            # Кэширование
            self.cache_manager.set(cache_key, summary)

            return summary

        except Exception as e:
            logging.error(f"Error summarizing text: {str(e)}")
            return f"Error summarizing text: {str(e)}"

    def paraphrase_text(self, text: str) -> str:
        """
        Перефразирование текста

        Args:
            text: Текст для обработки

        Returns:
            str: Перефразированный текст
        """
        try:
            if not text:
                return ""

            cache_key = self.cache_manager.generate_key(text, prefix="paraphrase")
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result

            # Ограничиваем длину входного текста
            max_input_length = 512
            if len(text) > max_input_length:
                text = text[:max_input_length]

            inputs = self.paraphraser_tokenizer(
                text,
                return_tensors="pt",
                max_length=max_input_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.paraphraser.generate(
                    **inputs, max_length=max_input_length, num_beams=4, temperature=1.0
                )

            paraphrased = self.paraphraser_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            # Кэширование
            self.cache_manager.set(cache_key, paraphrased)

            return paraphrased

        except Exception as e:
            logging.error(f"Error paraphrasing text: {str(e)}")
            return f"Error paraphrasing text: {str(e)}"

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Анализ текста

        Args:
            text: Текст для анализа

        Returns:
            Dict[str, Any]: Результаты анализа
        """
        try:
            if not text:
                return {"error": "Empty text provided"}

            cache_key = self.cache_manager.generate_key(text, prefix="analysis")
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result

            # Токенизация
            sentences = sent_tokenize(text)
            if not sentences:
                return {"error": "No sentences found in text"}

            words = word_tokenize(text)

            # Фильтрация стоп-слов
            meaningful_words = [
                word.lower()
                for word in words
                if word.lower() not in self.stop_words and word.isalnum()
            ]

            # Анализ сложности
            complexity = self._analyze_complexity(text)

            # Статистика
            analysis = {
                "sentence_count": len(sentences),
                "word_count": len(words),
                "unique_words": len(set(meaningful_words)),
                "avg_sentence_length": len(words) / max(len(sentences), 1),
                "complexity": complexity,
                "vocabulary_richness": len(set(meaningful_words)) / max(len(words), 1),
                "sentences": sentences[:3] if len(sentences) >= 3 else sentences,
            }

            # Кэширование
            self.cache_manager.set(cache_key, analysis)

            return analysis

        except Exception as e:
            logging.error(f"Error analyzing text: {str(e)}")
            return {"error": f"Error analyzing text: {str(e)}"}

    def _analyze_complexity(self, text: str) -> str:
        """
        Анализ сложности текста

        Args:
            text: Текст для анализа

        Returns:
            str: Уровень сложности
        """
        try:
            if not text:
                return "simple"

            # Ограничиваем длину входного текста
            max_input_length = 512
            if len(text) > max_input_length:
                text = text[:max_input_length]

            inputs = self.classifier_tokenizer(
                text,
                return_tensors="pt",
                max_length=max_input_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.classifier(**inputs)

            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)

            complexity_labels = ["simple", "medium", "complex"]
            return complexity_labels[prediction.item()]

        except Exception as e:
            logging.error(f"Error analyzing complexity: {str(e)}")
            return "medium"

    def cleanup(self):
        """Очистка ресурсов"""
        try:
            # Очистка GPU памяти
            if hasattr(self, "paraphraser") and self.paraphraser is not None:
                self.paraphraser = self.paraphraser.cpu()
            if hasattr(self, "classifier") and self.classifier is not None:
                self.classifier = self.classifier.cpu()
            torch.cuda.empty_cache()
            logging.info("TextProcessor resources cleaned up")
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
