"""
Модуль для генерации вопросов на основе текста.
Использует языковые модели для создания разнообразных вопросов.
"""

import logging
from typing import List, Dict, Optional, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
from .cache_manager import CacheManager
from .gpu_manager import GPUManager


class QuestionGenerator:
    def __init__(self, use_gpu: bool = True, max_text_length: int = 1024):
        """
        Инициализация генератора вопросов

        Args:
            use_gpu: Использовать ли GPU для обработки
            max_text_length: Максимальная длина текста для обработки
        """
        self.gpu_manager = GPUManager()
        self.device = self.gpu_manager.get_optimal_device(
            required_memory=2.0
        )  # Примерно 2GB для модели
        self.max_text_length = max_text_length

        # Инициализация моделей
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(
            self.device
        )
        self.cache_manager = CacheManager(cache_dir="./cache")

        logging.info(f"QuestionGenerator initialized using device: {self.device}")

    def _validate_text(self, text: str) -> Tuple[bool, str]:
        """
        Валидация входного текста

        Args:
            text: Текст для валидации

        Returns:
            Tuple[bool, str]: (Валидный ли текст, сообщение об ошибке)
        """
        if not text or not text.strip():
            return False, "Empty text provided"

        if len(text) > self.max_text_length:
            return False, f"Text too long (max {self.max_text_length} characters)"

        # Проверка на минимальное количество слов
        if len(text.split()) < 10:
            return False, "Text too short (min 10 words required)"

        return True, ""

    def generate_questions(
        self,
        text: str,
        num_questions: int = 3,
        question_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Генерация вопросов на основе текста

        Args:
            text: Текст для генерации вопросов
            num_questions: Количество вопросов для генерации
            question_types: Типы вопросов для генерации

        Returns:
            List[Dict]: Список сгенерированных вопросов
        """
        try:
            # Валидация текста
            is_valid, error_message = self._validate_text(text)
            if not is_valid:
                logging.error(f"Text validation failed: {error_message}")
                return []

            # Проверяем кэш
            cache_key = self.cache_manager._generate_key(text, prefix="questions")
            cached_questions = self.cache_manager.get(cache_key, "questions")
            if cached_questions:
                return cached_questions[:num_questions]

            if question_types is None:
                question_types = ["multiple_choice", "true_false", "open_ended"]

            questions = []
            for q_type in question_types:
                if len(questions) >= num_questions:
                    break

                try:
                    if q_type == "multiple_choice":
                        new_questions = self._generate_multiple_choice(text)
                    elif q_type == "true_false":
                        new_questions = self._generate_true_false(text)
                    elif q_type == "open_ended":
                        new_questions = self._generate_open_ended(text)
                    else:
                        continue

                    if new_questions:
                        questions.extend(new_questions)
                except Exception as e:
                    logging.error(f"Error generating {q_type} questions: {str(e)}")
                    continue

            # Если не удалось сгенерировать вопросы
            if not questions:
                logging.warning("Failed to generate any questions")
                return []

            # Добавляем объяснения к каждому вопросу
            questions = [self._add_explanation(q, text) for q in questions]

            # Сохраняем в кэш
            self.cache_manager.set(cache_key, questions, "questions")

            return questions[:num_questions]

        except Exception as e:
            logging.error(f"Error generating questions: {str(e)}")
            return []
        finally:
            # Очищаем память GPU
            self.gpu_manager.cleanup()

    def _generate_multiple_choice(self, text: str) -> List[Dict]:
        """Генерация вопросов с множественным выбором"""
        prompt = f"Generate a multiple choice question based on this text: {text}"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        outputs = self.model.generate(
            input_ids,
            max_length=150,
            num_beams=4,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
        )

        question_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if not question_text:
            return []

        # Генерация вариантов ответов
        options = self._generate_answer_options(text, question_text)
        if not options:
            return []

        return [
            {
                "type": "multiple_choice",
                "question": question_text,
                "options": options,
                "correct_answer": options[0],  # Первый вариант - правильный
                "difficulty": self._calculate_difficulty(text, question_text),
            }
        ]

    def _generate_true_false(self, text: str) -> List[Dict]:
        """Генерация вопросов типа правда/ложь"""
        prompt = f"Generate a true/false question based on this text: {text}"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        outputs = self.model.generate(
            input_ids,
            max_length=100,
            num_beams=4,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        question_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if not question_text:
            return []

        is_true = np.random.choice([True, False])
        if not is_true:
            question_text = self._negate_statement(question_text)

        return [
            {
                "type": "true_false",
                "question": question_text,
                "correct_answer": is_true,
                "difficulty": self._calculate_difficulty(text, question_text),
            }
        ]

    def _generate_open_ended(self, text: str) -> List[Dict]:
        """Генерация открытых вопросов"""
        prompt = f"Generate an open-ended question based on this text: {text}"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        outputs = self.model.generate(
            input_ids,
            max_length=100,
            num_beams=4,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        question_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if not question_text:
            return []

        answer = self._generate_answer(text, question_text)
        if not answer:
            return []

        return [
            {
                "type": "open_ended",
                "question": question_text,
                "correct_answer": answer,
                "difficulty": self._calculate_difficulty(text, question_text),
            }
        ]

    def _add_explanation(self, question: Dict, text: str) -> Dict:
        """Добавление объяснения к вопросу"""
        try:
            prompt = f"Explain why this is the correct answer. Text: {text}\nQuestion: {question['question']}\nAnswer: {question['correct_answer']}"
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                self.device
            )

            outputs = self.model.generate(
                input_ids,
                max_length=150,
                num_beams=4,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

            explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            question["explanation"] = explanation

        except Exception as e:
            logging.error(f"Error generating explanation: {str(e)}")
            question["explanation"] = "No explanation available"

        return question

    def _calculate_difficulty(self, text: str, question: str) -> str:
        """Оценка сложности вопроса"""
        try:
            # Простая эвристика на основе длины вопроса и сложности слов
            words = question.split()
            avg_word_length = sum(len(word) for word in words) / len(words)

            if avg_word_length > 8:
                return "hard"
            elif avg_word_length > 6:
                return "medium"
            else:
                return "easy"
        except:
            return "medium"

    def _negate_statement(self, statement: str) -> str:
        """Преобразование утверждения в отрицание"""
        negations = {
            "is": "is not",
            "are": "are not",
            "was": "was not",
            "were": "were not",
            "will": "will not",
            "has": "has not",
            "have": "have not",
            "can": "cannot",
            "should": "should not",
        }

        for word, negation in negations.items():
            if f" {word} " in statement:
                return statement.replace(f" {word} ", f" {negation} ")

        return f"It is not true that {statement.lower()}"

    def _generate_answer_options(self, text: str, question: str) -> List[str]:
        """Генерация вариантов ответов для вопроса с множественным выбором"""
        prompt = f"Generate 4 answer options for this question based on the text: {text}\nQuestion: {question}"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        outputs = self.model.generate(
            input_ids,
            max_length=200,
            num_beams=4,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        options_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        options = [opt.strip() for opt in options_text.split("\n") if opt.strip()]

        # Если не удалось сгенерировать 4 варианта, добавляем случайные
        while len(options) < 4:
            options.append(f"Option {len(options) + 1}")

        return options[:4]

    def _generate_answer(self, text: str, question: str) -> str:
        """Генерация ответа на открытый вопрос"""
        prompt = f"Answer this question based on the text: {text}\nQuestion: {question}"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        outputs = self.model.generate(
            input_ids,
            max_length=200,
            num_beams=4,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
