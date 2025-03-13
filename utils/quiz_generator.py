"""
Модуль для генерации интерактивных тестов.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import random
from .question_generator import QuestionGenerator
from .cache_manager import CacheManager


@dataclass
class Quiz:
    """Структура данных для хранения теста"""

    quiz_id: str
    title: str
    description: str
    questions: List[Dict[str, Any]]
    difficulty: float
    time_limit: int  # в секундах
    tags: List[str]


class QuizGenerator:
    def __init__(self, use_gpu: bool = True):
        """
        Инициализация генератора тестов

        Args:
            use_gpu: Использовать ли GPU для обработки
        """
        self.question_generator = QuestionGenerator(use_gpu=use_gpu)
        self.cache_manager = CacheManager(cache_dir="./cache/quizzes")

        logging.info("QuizGenerator initialized")

    def generate_quiz(
        self,
        content: str,
        title: str,
        num_questions: int = 5,
        difficulty: float = 1.0,
        time_limit: int = 600,
        tags: Optional[List[str]] = None,
    ) -> Quiz:
        """
        Генерация теста на основе контента

        Args:
            content: Исходный контент
            title: Название теста
            num_questions: Количество вопросов
            difficulty: Уровень сложности (0.1-2.0)
            time_limit: Ограничение по времени в секундах
            tags: Теги для теста

        Returns:
            Quiz: Сгенерированный тест
        """
        try:
            # Проверка кэша
            cache_key = self.cache_manager.generate_key(
                f"{content}_{num_questions}_{difficulty}", prefix="quiz"
            )
            cached_quiz = self.cache_manager.get(cache_key)
            if cached_quiz:
                return self._deserialize_quiz(cached_quiz)

            # Генерация вопросов
            questions = self.question_generator.generate_questions(
                content, num_questions=num_questions, difficulty=difficulty
            )

            # Создание теста
            quiz = Quiz(
                quiz_id=cache_key,
                title=title,
                description=self._generate_description(content),
                questions=questions,
                difficulty=difficulty,
                time_limit=time_limit,
                tags=tags or [],
            )

            # Кэширование
            self._save_quiz(quiz)

            logging.info(f"Generated quiz '{title}' with {num_questions} questions")
            return quiz

        except Exception as e:
            logging.error(f"Error generating quiz: {str(e)}")
            raise

    def get_quiz(self, quiz_id: str) -> Optional[Quiz]:
        """
        Получение теста по ID

        Args:
            quiz_id: ID теста

        Returns:
            Optional[Quiz]: Тест, если найден
        """
        try:
            cached_quiz = self.cache_manager.get(quiz_id)
            if cached_quiz:
                return self._deserialize_quiz(cached_quiz)
            return None

        except Exception as e:
            logging.error(f"Error getting quiz: {str(e)}")
            raise

    def grade_quiz(self, quiz: Quiz, answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оценка ответов на тест

        Args:
            quiz: Тест для оценки
            answers: Ответы пользователя

        Returns:
            Dict[str, Any]: Результаты оценки
        """
        try:
            total_questions = len(quiz.questions)
            correct_answers = 0
            question_results = []

            for question in quiz.questions:
                question_id = question["id"]
                if question_id not in answers:
                    continue

                user_answer = answers[question_id]
                is_correct = self._check_answer(question, user_answer)

                if is_correct:
                    correct_answers += 1

                question_results.append(
                    {
                        "question_id": question_id,
                        "correct": is_correct,
                        "user_answer": user_answer,
                        "correct_answer": question["answer"],
                        "explanation": question.get("explanation", ""),
                    }
                )

            score = (
                (correct_answers / total_questions) * 100 if total_questions > 0 else 0
            )

            return {
                "quiz_id": quiz.quiz_id,
                "total_questions": total_questions,
                "correct_answers": correct_answers,
                "score": score,
                "question_results": question_results,
                "passed": score >= 70,  # Проходной балл 70%
            }

        except Exception as e:
            logging.error(f"Error grading quiz: {str(e)}")
            raise

    def _generate_description(self, content: str) -> str:
        """
        Генерация описания теста

        Args:
            content: Исходный контент

        Returns:
            str: Описание теста
        """
        try:
            # Здесь можно добавить более сложную логику генерации описания
            return f"Тест на основе предоставленного материала. Содержит вопросы разных типов."

        except Exception as e:
            logging.error(f"Error generating description: {str(e)}")
            return "Тест для проверки знаний"

    def _check_answer(self, question: Dict[str, Any], user_answer: Any) -> bool:
        """
        Проверка правильности ответа

        Args:
            question: Вопрос с правильным ответом
            user_answer: Ответ пользователя

        Returns:
            bool: True, если ответ правильный
        """
        try:
            question_type = question.get("type", "multiple_choice")
            correct_answer = question["answer"]

            if question_type == "multiple_choice":
                return user_answer == correct_answer
            elif question_type == "text":
                # Для текстовых ответов можно добавить более сложную логику сравнения
                return user_answer.lower().strip() == correct_answer.lower().strip()
            elif question_type == "multiple_select":
                return set(user_answer) == set(correct_answer)

            return False

        except Exception as e:
            logging.error(f"Error checking answer: {str(e)}")
            return False

    def _save_quiz(self, quiz: Quiz):
        """
        Сохранение теста в кэш

        Args:
            quiz: Тест для сохранения
        """
        try:
            quiz_data = {
                "quiz_id": quiz.quiz_id,
                "title": quiz.title,
                "description": quiz.description,
                "questions": quiz.questions,
                "difficulty": quiz.difficulty,
                "time_limit": quiz.time_limit,
                "tags": quiz.tags,
            }

            self.cache_manager.set(quiz.quiz_id, quiz_data)

        except Exception as e:
            logging.error(f"Error saving quiz: {str(e)}")
            raise

    def _deserialize_quiz(self, data: Dict) -> Quiz:
        """
        Десериализация теста из кэша

        Args:
            data: Сериализованные данные

        Returns:
            Quiz: Десериализованный тест
        """
        try:
            return Quiz(
                quiz_id=data["quiz_id"],
                title=data["title"],
                description=data["description"],
                questions=data["questions"],
                difficulty=data["difficulty"],
                time_limit=data["time_limit"],
                tags=data["tags"],
            )
        except Exception as e:
            logging.error(f"Error deserializing quiz: {str(e)}")
            raise
