"""
Модуль для обработки различных типов контента.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import whisper
import yt_dlp
import cv2
import numpy as np
from transformers import pipeline
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize
import torch
import fitz  # PyMuPDF для PDF
import requests
from bs4 import BeautifulSoup
import tempfile
from urllib.parse import urlparse
from .question_generator import QuestionGenerator
from .concept_extractor import ConceptExtractor
from .cache_manager import CacheManager
from .result_printer import print_processing_results, ContentType
import openai_whisper as whisper


@dataclass
class ProcessedContent:
    """Структура данных для хранения обработанного контента"""

    content_type: str
    original_content: str
    processed_content: Dict[str, Any]
    summary: str
    key_points: List[str]
    concepts: List[Dict[str, str]]
    tags: List[str]
    interactive_elements: Dict[str, Any]
    metadata: Dict[str, Any]


class ContentProcessor:
    def __init__(self, cache_dir: str = "./cache", use_gpu: bool = True):
        """
        Инициализация процессора контента

        Args:
            cache_dir: Директория для кэширования
            use_gpu: Использовать ли GPU для обработки
        """
        self.cache_dir = cache_dir
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Инициализация моделей
        self.whisper_model = whisper.load_model("base", device=self.device)
        self.summarizer = pipeline(
            "summarization", model="facebook/bart-large-cnn", device=self.device
        )
        self.question_generator = QuestionGenerator(use_gpu=use_gpu)
        self.concept_extractor = ConceptExtractor()
        self.cache_manager = CacheManager(cache_dir=cache_dir)

        # Создаем директории для кэша
        os.makedirs(cache_dir, exist_ok=True)
        for dir_name in ["videos", "audio", "text", "pdf", "images", "web"]:
            os.makedirs(os.path.join(cache_dir, dir_name), exist_ok=True)

        logging.info(f"ContentProcessor initialized using device: {self.device}")

    async def process_content(
        self, content: str, content_type: str = "video", title: str = None
    ) -> Dict[str, Any]:
        """
        Обработка контента в зависимости от его типа

        Args:
            content: URL или текст для обработки
            content_type: Тип контента ("video", "text", "audio", "youtube", "pdf", "web", "image")
            title: Опциональное название контента

        Returns:
            Dict с результатами обработки
        """
        try:
            # Автоопределение типа контента, если не указан
            if content_type == "auto":
                content_type = self._detect_content_type(content)

            # Проверяем кэш
            cache_key = self.cache_manager.generate_key(content, prefix=content_type)
            cached_result = self.cache_manager.get(cache_key, content_type)
            if cached_result:
                print_processing_results(
                    cached_result, getattr(ContentType, content_type.upper())
                )
                return cached_result

            # Обработка в зависимости от типа
            if content_type in ["video", "youtube"]:
                result = await self._process_video(content, title)
            elif content_type == "text":
                result = await self._process_text(content, title)
            elif content_type == "audio":
                result = await self._process_audio(content, title)
            elif content_type == "pdf":
                result = await self._process_pdf(content, title)
            elif content_type == "web":
                result = await self._process_web(content, title)
            elif content_type == "image":
                result = await self._process_image(content, title)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")

            # Кэшируем результат
            if result["status"] == "success":
                self.cache_manager.set(cache_key, result, content_type)

            # Выводим результаты
            print_processing_results(result, getattr(ContentType, content_type.upper()))
            return result

        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                f"{content_type}_info": {"source": content},
            }
            print_processing_results(
                error_result, getattr(ContentType, content_type.upper())
            )
            return error_result

    def _detect_content_type(self, content: str) -> str:
        """Автоопределение типа контента"""
        # Проверяем URL
        if content.startswith(("http://", "https://")):
            parsed_url = urlparse(content)
            # YouTube
            if "youtube.com" in parsed_url.netloc or "youtu.be" in parsed_url.netloc:
                return "youtube"
            # PDF
            elif content.lower().endswith(".pdf"):
                return "pdf"
            # Изображения
            elif any(
                content.lower().endswith(ext)
                for ext in [".jpg", ".jpeg", ".png", ".gif"]
            ):
                return "image"
            # Аудио
            elif any(content.lower().endswith(ext) for ext in [".mp3", ".wav", ".ogg"]):
                return "audio"
            # Видео
            elif any(content.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov"]):
                return "video"
            # Веб-страница
            else:
                return "web"
        # Локальные файлы
        elif os.path.exists(content):
            ext = os.path.splitext(content)[1].lower()
            if ext == ".pdf":
                return "pdf"
            elif ext in [".jpg", ".jpeg", ".png", ".gif"]:
                return "image"
            elif ext in [".mp3", ".wav", ".ogg"]:
                return "audio"
            elif ext in [".mp4", ".avi", ".mov"]:
                return "video"
            elif ext in [".txt", ".doc", ".docx"]:
                return "text"
        # Если ничего не подошло, считаем текстом
        return "text"

    async def _process_video(self, url: str, title: str = None) -> Dict[str, Any]:
        """Обработка видео контента"""
        try:
            # Загрузка видео
            video_path = await self._download_video(url)

            # Извлечение аудио
            audio_path = self._extract_audio(video_path)

            # Транскрибация
            transcript = self._transcribe_audio(audio_path)

            # Сегментация
            segments = self._segment_content(transcript["text"])

            # Обработка сегментов
            processed_segments = []
            for segment in segments:
                # Создаем ключевые моменты
                key_points = self._extract_key_points(segment["text"])

                # Генерируем вопросы
                questions = self._generate_questions(segment["text"])

                # Извлекаем концепции
                concepts = self._extract_concepts(segment["text"])

                # Создаем thumbnail для сегмента
                thumbnail = self._create_thumbnail(video_path, segment["start_time"])

                processed_segments.append(
                    {
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                        "text": segment["text"],
                        "key_points": key_points,
                        "concepts": concepts,
                        "thumbnail": thumbnail,
                        "interactive_elements": {"questions": questions},
                    }
                )

            # Получаем метаданные видео
            metadata = {
                "duration": self._get_video_duration(video_path),
                "resolution": self._get_video_resolution(video_path),
                "processed_on": self.device,
                "original_url": url,
            }

            # Очистка временных файлов
            self._cleanup_temp_files(video_path, audio_path)

            return {
                "status": "success",
                "segments": processed_segments,
                "metadata": metadata,
            }

        except Exception as e:
            logging.error(f"Error processing video: {str(e)}")
            raise

    async def _process_text(self, text: str, title: str = None) -> Dict[str, Any]:
        """Обработка текстового контента"""
        try:
            # Очистка текста
            cleaned_text = self._clean_text(text)

            # Сегментация
            segments = self._segment_content(cleaned_text)

            # Обработка сегментов
            processed_segments = []
            for segment in segments:
                # Упрощение текста
                simplified = self._simplify_text(segment["text"])

                # Извлечение ключевых моментов
                key_points = self._extract_key_points(segment["text"])

                # Генерация вопросов
                questions = self._generate_questions(segment["text"])

                # Извлечение концепций
                concepts = self._extract_concepts(segment["text"])

                processed_segments.append(
                    {
                        "text": segment["text"],
                        "simplified_text": simplified,
                        "key_points": key_points,
                        "concepts": concepts,
                        "interactive_elements": {"questions": questions},
                    }
                )

            return {
                "status": "success",
                "segments": processed_segments,
                "metadata": {"text_length": len(text), "processed_on": self.device},
            }

        except Exception as e:
            logging.error(f"Error processing text: {str(e)}")
            raise

    async def _download_video(self, url: str) -> str:
        """Загрузка видео"""
        output_path = os.path.join(self.cache_dir, "videos", "temp_video.mp4")
        ydl_opts = {
            "format": "best[ext=mp4]",
            "outtmpl": output_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path

    def _extract_audio(self, video_path: str) -> str:
        """Извлечение аудио из видео"""
        audio_path = os.path.join(self.cache_dir, "audio", "temp_audio.wav")
        video = AudioSegment.from_file(video_path)
        video.export(audio_path, format="wav")
        return audio_path

    def _transcribe_audio(self, audio_path: str) -> Dict:
        """Транскрибация аудио"""
        return self.whisper_model.transcribe(audio_path)

    def _segment_content(self, text: str) -> List[Dict]:
        """Сегментация контента на смысловые части"""
        segments = []
        sentences = sent_tokenize(text)

        current_segment = {"text": "", "start": 0}
        word_count = 0

        for sentence in sentences:
            words = sentence.split()
            word_count += len(words)

            if word_count > 100:  # Новый сегмент каждые ~100 слов
                segments.append(current_segment)
                current_segment = {"text": sentence, "start": len(segments)}
                word_count = len(words)
            else:
                current_segment["text"] += " " + sentence

        if current_segment["text"]:
            segments.append(current_segment)

        return segments

    def _extract_key_points(self, text: str) -> List[str]:
        """Извлечение ключевых моментов"""
        summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
        return sent_tokenize(summary[0]["summary_text"])

    def _generate_questions(self, text: str) -> List[Dict]:
        """Генерация вопросов"""
        return self.question_generator.generate_questions(
            text, num_questions=3, question_types=["multiple_choice", "true_false"]
        )

    def _extract_concepts(self, text: str) -> List[Dict]:
        """Извлечение концепций"""
        return self.concept_extractor.extract_concepts(text)

    def _create_thumbnail(self, video_path: str, timestamp: float) -> Optional[str]:
        """Создание thumbnail для сегмента видео"""
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()

            if ret:
                thumbnail_path = os.path.join(
                    self.cache_dir, "videos", f"thumbnail_{timestamp}.jpg"
                )
                cv2.imwrite(thumbnail_path, frame)
                cap.release()
                return thumbnail_path

            cap.release()
            return None
        except Exception as e:
            logging.warning(f"Error creating thumbnail: {str(e)}")
            return None

    def _get_video_duration(self, video_path: str) -> float:
        """Получение длительности видео"""
        cap = cv2.VideoCapture(video_path)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return duration

    def _get_video_resolution(self, video_path: str) -> tuple:
        """Получение разрешения видео"""
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (width, height)

    def _cleanup_temp_files(self, *file_paths: str):
        """Очистка временных файлов"""
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logging.warning(f"Error cleaning up file {path}: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """Очистка текста"""
        return " ".join(text.split())

    def _simplify_text(self, text: str) -> str:
        """Упрощение текста"""
        summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]["summary_text"]

    async def _process_pdf(self, source: str, title: str = None) -> Dict[str, Any]:
        """Обработка PDF документов"""
        try:
            # Загрузка PDF
            if source.startswith(("http://", "https://")):
                # Скачиваем PDF
                response = requests.get(source)
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as temp_file:
                    temp_file.write(response.content)
                    pdf_path = temp_file.name
            else:
                pdf_path = source

            # Открываем PDF
            doc = fitz.open(pdf_path)

            # Извлекаем текст и изображения
            processed_segments = []
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Извлекаем текст
                text = page.get_text()
                if not text.strip():
                    continue

                # Обрабатываем текст
                segments = self._segment_content(text)

                for segment in segments:
                    # Упрощение текста
                    simplified = self._simplify_text(segment["text"])

                    # Извлечение ключевых моментов
                    key_points = self._extract_key_points(segment["text"])

                    # Генерация вопросов
                    questions = self._generate_questions(segment["text"])

                    # Извлечение концепций
                    concepts = self._extract_concepts(segment["text"])

                    processed_segments.append(
                        {
                            "page": page_num + 1,
                            "text": segment["text"],
                            "simplified_text": simplified,
                            "key_points": key_points,
                            "concepts": concepts,
                            "interactive_elements": {"questions": questions},
                        }
                    )

            # Получаем метаданные
            metadata = {
                "title": doc.metadata.get("title", title),
                "author": doc.metadata.get("author"),
                "pages": len(doc),
                "processed_on": self.device,
            }

            # Закрываем документ
            doc.close()

            # Удаляем временный файл, если это был URL
            if source.startswith(("http://", "https://")):
                os.unlink(pdf_path)

            return {
                "status": "success",
                "segments": processed_segments,
                "metadata": metadata,
            }

        except Exception as e:
            logging.error(f"Error processing PDF: {str(e)}")
            raise

    async def _process_web(self, url: str, title: str = None) -> Dict[str, Any]:
        """Обработка веб-страниц"""
        try:
            # Загружаем страницу
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # Парсим HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Удаляем ненужные элементы
            for tag in soup(["script", "style", "nav", "footer", "iframe"]):
                tag.decompose()

            # Извлекаем основной контент
            title = title or soup.title.string if soup.title else url
            main_content = ""

            # Пытаемся найти основной контент
            for tag in ["article", "main", "div.content", "div.main"]:
                content = soup.select_one(tag)
                if content:
                    main_content = content.get_text(separator="\n", strip=True)
                    break

            # Если не нашли, берем весь текст
            if not main_content:
                main_content = soup.get_text(separator="\n", strip=True)

            # Обрабатываем текст
            result = await self._process_text(main_content, title)

            # Добавляем метаданные веб-страницы
            result["metadata"].update(
                {
                    "url": url,
                    "title": title,
                    "description": (
                        soup.find("meta", {"name": "description"})["content"]
                        if soup.find("meta", {"name": "description"})
                        else None
                    ),
                    "keywords": (
                        soup.find("meta", {"name": "keywords"})["content"]
                        if soup.find("meta", {"name": "keywords"})
                        else None
                    ),
                }
            )

            return result

        except Exception as e:
            logging.error(f"Error processing web page: {str(e)}")
            raise

    async def _process_image(self, source: str, title: str = None) -> Dict[str, Any]:
        """Обработка изображений"""
        try:
            # Загрузка изображения
            if source.startswith(("http://", "https://")):
                response = requests.get(source)
                with tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(source)[1], delete=False
                ) as temp_file:
                    temp_file.write(response.content)
                    image_path = temp_file.name
            else:
                image_path = source

            # Открываем изображение
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")

            # Извлекаем текст (если есть)
            try:
                import pytesseract

                text = pytesseract.image_to_string(image)
            except:
                text = ""

            # Создаем сегмент
            segment = {
                "image_path": image_path,
                "text": text,
                "dimensions": {"width": image.shape[1], "height": image.shape[0]},
            }

            # Если есть текст, обрабатываем его
            if text.strip():
                # Извлечение ключевых моментов
                segment["key_points"] = self._extract_key_points(text)

                # Генерация вопросов
                segment["interactive_elements"] = {
                    "questions": self._generate_questions(text)
                }

                # Извлечение концепций
                segment["concepts"] = self._extract_concepts(text)

            # Удаляем временный файл, если это был URL
            if source.startswith(("http://", "https://")):
                os.unlink(image_path)

            return {
                "status": "success",
                "segments": [segment],
                "metadata": {
                    "dimensions": segment["dimensions"],
                    "has_text": bool(text.strip()),
                    "processed_on": self.device,
                },
            }

        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            raise
