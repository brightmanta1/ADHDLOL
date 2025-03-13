import json
import os
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, time
import asyncio
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pinecone
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
import networkx as nx
from sklearn.cluster import KMeans
import whisper
import cv2
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import re
import math


@dataclass
class ContentModule:
    """Структура модуля контента."""

    title: str
    subtopics: List[str]
    difficulty_level: float
    prerequisites: List[str]
    learning_objectives: List[str]
    estimated_time: int
    tags: List[str]
    visual_aids: List[Dict[str, Any]]
    simplified_version: Optional[Dict[str, Any]] = None


@dataclass
class UserConcentrationMetrics:
    """Метрики концентрации пользователя."""

    time_of_day: time
    duration: int
    focus_score: float
    interruption_count: int
    completion_rate: float
    engagement_level: float
    environment_factors: Dict[str, Any]


@dataclass
class LearningContent:
    """Структура для хранения обработанного контента."""

    title: str
    content: str
    content_type: str
    summary: str
    key_points: List[str]
    concepts: List[Dict[str, Any]]
    difficulty: str
    visualizations: List[Dict[str, Any]]
    exercises: List[Dict[str, Any]]
    learning_path: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processed_at: datetime
    engagement_metrics: Dict[str, float] = None
    adaptive_difficulty: float = 0.5
    learning_progress: Dict[str, Any] = None
    modules: List[ContentModule] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    visual_code: Dict[str, Any] = field(default_factory=dict)
    smart_categories: List[Dict[str, Any]] = field(default_factory=list)


class PersonalizationEngine:
    """Движок персонализации на основе Pinecone."""

    def __init__(self, pinecone_index):
        self.index = pinecone_index
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.namespace_users = "user_profiles"
        self.namespace_patterns = "learning_patterns"
        self.namespace_content = "learning_content"

    async def update_user_profile(self, user_id: str, interaction_data: Dict[str, Any]):
        """Обновление профиля пользователя в Pinecone."""
        try:
            interaction_text = self._prepare_interaction_text(interaction_data)
            embedding = self.embedding_model.encode(interaction_text)
            metadata = {
                "last_interaction": datetime.now().isoformat(),
                "learning_style": interaction_data.get("learning_style", ""),
                "difficulty_preference": interaction_data.get("difficulty", 0.5),
                "topics_of_interest": json.dumps(interaction_data.get("topics", [])),
                "successful_patterns": json.dumps(
                    interaction_data.get("successful_patterns", [])
                ),
                "concentration_times": json.dumps(
                    interaction_data.get("concentration_times", [])
                ),
                "preferred_content_types": json.dumps(
                    interaction_data.get("preferred_content_types", [])
                ),
            }
            self.index.upsert(
                vectors=[(f"user_{user_id}", embedding.tolist(), metadata)],
                namespace=self.namespace_users,
            )
        except Exception as e:
            print(f"Ошибка при обновлении профиля: {str(e)}")

    async def get_optimal_learning_time(self, user_id: str) -> Dict[str, Any]:
        """Определение оптимального времени для обучения."""
        try:
            patterns = self.index.fetch(
                ids=[f"pattern_{user_id}_*"], namespace=self.namespace_patterns
            )
            if not patterns.vectors:
                return {}
            success_patterns = [
                json.loads(v.metadata["context"])
                for v in patterns.vectors.values()
                if float(v.metadata["success_rate"]) > 0.7
            ]
            time_analysis = self._analyze_optimal_times(success_patterns)
            return {
                "optimal_times": time_analysis["optimal_times"],
                "optimal_duration": time_analysis["optimal_duration"],
                "recommended_breaks": time_analysis["recommended_breaks"],
                "confidence": time_analysis["confidence"],
            }
        except Exception as e:
            print(f"Ошибка при определении оптимального времени: {str(e)}")
            return {}

    def _analyze_optimal_times(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ оптимального времени на основе успешных паттернов."""
        if not patterns:
            return {
                "optimal_times": [],
                "optimal_duration": 0,
                "recommended_breaks": [],
                "confidence": 0.0,
            }
        times = [p["time_of_day"] for p in patterns]
        durations = [p["duration"] for p in patterns]
        breaks = [p.get("breaks", []) for p in patterns]

        time_values = [[self._time_to_minutes(t)] for t in times]
        kmeans = KMeans(n_clusters=min(3, len(time_values)))
        kmeans.fit(time_values)

        optimal_times = []
        for center in kmeans.cluster_centers_:
            optimal_times.append(self._minutes_to_time(center[0]))

        return {
            "optimal_times": sorted(optimal_times),
            "optimal_duration": int(np.mean(durations)),
            "recommended_breaks": self._analyze_breaks(breaks),
            "confidence": self._calculate_confidence(patterns),
        }


class LearningScheduler:
    """Планировщик обучения."""

    def __init__(self, personalization_engine: PersonalizationEngine):
        self.personalization = personalization_engine
        self.schedule_preferences = {}

    async def create_personalized_schedule(self, user_id: str) -> Dict[str, Any]:
        """Создание персонализированного расписания."""
        try:
            schedule = self._create_concentration_schedule(user_id)
            optimal_time = await self.personalization.get_optimal_learning_time(user_id)

            if user_id in self.schedule_preferences:
                schedule = self._adjust_schedule_for_preferences(
                    schedule, self.schedule_preferences[user_id]
                )

            effectiveness = self._calculate_schedule_effectiveness(user_id, schedule)

            return {
                "daily_schedule": schedule["daily_schedule"],
                "optimal_learning_times": optimal_time["optimal_times"],
                "recommended_breaks": schedule["recommended_breaks"],
                "environment_recommendations": schedule["environment_recommendations"],
                "effectiveness_score": effectiveness,
                "visualization": self._generate_schedule_visualization(schedule),
            }
        except Exception as e:
            print(f"Ошибка при создании расписания: {str(e)}")
            return {}

    def update_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """Обновление предпочтений пользователя."""
        self.schedule_preferences[user_id] = {
            "preferred_times": preferences.get("preferred_times", []),
            "max_session_duration": preferences.get("max_session_duration", 60),
            "min_break_duration": preferences.get("min_break_duration", 5),
            "preferred_activity_types": preferences.get("preferred_activity_types", []),
            "environmental_preferences": preferences.get(
                "environmental_preferences", {}
            ),
            "updated_at": datetime.now(),
        }

    def _create_concentration_schedule(self, user_id: str) -> Dict[str, Any]:
        """Создание базового расписания."""
        metrics = self._analyze_concentration_patterns(user_id)
        optimal_times = [
            {
                "time": m.time_of_day,
                "duration": m.duration,
                "score": m.focus_score * m.engagement_level,
            }
            for m in metrics
            if m.focus_score > 0.7 and m.engagement_level > 0.6
        ]

        return {
            "daily_schedule": self._generate_daily_schedule(optimal_times),
            "recommended_breaks": self._calculate_break_intervals(metrics),
            "environment_recommendations": self._analyze_environment_factors(metrics),
        }

    def _generate_daily_schedule(
        self, optimal_times: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Генерация дневного расписания."""
        schedule = []
        total_duration = 0
        max_daily_duration = 240  # 4 часа максимум

        for time_slot in sorted(optimal_times, key=lambda x: x["score"], reverse=True):
            if total_duration >= max_daily_duration:
                break

            duration = min(time_slot["duration"], 60)
            schedule.append(
                {
                    "start_time": time_slot["time"],
                    "duration": duration,
                    "activity_type": self._recommend_activity_type(time_slot["score"]),
                    "break_duration": max(5, duration // 4),
                }
            )
            total_duration += duration

        return schedule

    def _generate_schedule_visualization(self, schedule: Dict[str, Any]) -> go.Figure:
        """Создание визуализации расписания."""
        try:
            times = []
            durations = []
            activities = []
            breaks = []

            for slot in schedule["daily_schedule"]:
                times.append(slot["start_time"].strftime("%H:%M"))
                durations.append(slot["duration"])
                activities.append(slot["activity_type"])
                breaks.append(slot["break_duration"])

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    name="Длительность занятия",
                    x=times,
                    y=durations,
                    text=activities,
                    textposition="auto",
                )
            )

            fig.add_trace(
                go.Bar(
                    name="Длительность перерыва",
                    x=times,
                    y=breaks,
                    text=[f"{b} мин" for b in breaks],
                    textposition="auto",
                )
            )

            fig.update_layout(
                title="Дневное расписание",
                xaxis_title="Время",
                yaxis_title="Длительность (минуты)",
                barmode="group",
                showlegend=True,
            )

            return fig
        except Exception as e:
            print(f"Ошибка при создании визуализации: {str(e)}")
            return None


class ContentProcessor:
    """Процессор для обработки и улучшения различных типов контента."""

    def __init__(self):
        self.whisper_model = whisper.load_model("base")
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.pomodoro_settings = {
            "work_duration": 25,
            "break_duration": 5,
            "long_break_duration": 15,
            "sessions_before_long_break": 4,
        }

    async def process_content(
        self, content: str, content_type: str, title: str = None
    ) -> Dict[str, Any]:
        """Обработка и улучшение контента."""
        # Базовая обработка
        base_result = await self._process_by_type(content, content_type, title)

        # Улучшение контента
        enhanced_content = await self._enhance_content(base_result)

        # Создание мультисенсорных элементов
        multisensory = await self._create_multisensory_elements(enhanced_content)

        # Генерация интерактивных элементов
        interactive = await self._generate_interactive_elements(enhanced_content)

        return {
            **enhanced_content,
            "multisensory_elements": multisensory,
            "interactive_elements": interactive,
            "learning_aids": {
                "infographics": enhanced_content["infographics"],
                "mind_maps": enhanced_content["mind_maps"],
                "structured_notes": enhanced_content["structured_notes"],
                "pomodoro_schedule": self._create_pomodoro_schedule(enhanced_content),
                "quick_references": enhanced_content["quick_references"],
            },
        }

    async def _process_by_type(
        self, content: str, content_type: str, title: str = None
    ) -> Dict[str, Any]:
        """Обработка контента в зависимости от типа."""
        processors = {
            "text": self._process_text,
            "audio": self._process_audio,
            "video": self._process_video,
            "image": self._process_image,
            "pdf": self._process_pdf,
            "youtube": self._process_youtube,
            "website": self._process_website,
        }

        processor = processors.get(content_type.lower())
        if not processor:
            raise ValueError(f"Неподдерживаемый тип контента: {content_type}")

        return await processor(content, title)

    async def _enhance_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Улучшение контента для СДВГ-дружественного формата."""
        enhanced = {
            **content,
            "structured_notes": self._create_structured_notes(content["content"]),
            "infographics": await self._generate_infographics(content),
            "mind_maps": self._create_mind_maps(content),
            "quick_references": self._create_quick_references(content),
            "visual_hierarchy": self._create_visual_hierarchy(content),
            "smart_tags": self._generate_smart_tags(content),
            "categories": self._categorize_content(content),
        }

        # Добавление цветового кодирования
        enhanced["color_coded_sections"] = self._apply_color_coding(
            enhanced["structured_notes"]
        )

        # Создание иерархической структуры
        enhanced["hierarchy"] = {
            "main_topics": self._extract_main_topics(content),
            "subtopics": self._extract_subtopics(content),
            "relationships": self._identify_relationships(content),
        }

        return enhanced

    def _create_structured_notes(self, text: str) -> List[Dict[str, Any]]:
        """Преобразование текста в структурированные заметки."""
        sections = []
        current_section = {"title": "", "content": [], "type": "main"}

        for line in text.split("\n"):
            # Определение типа строки (заголовок, подзаголовок, список и т.д.)
            line_type = self._determine_line_type(line)

            if line_type == "header":
                if current_section["title"]:
                    sections.append(current_section)
                current_section = {"title": line.strip(), "content": [], "type": "main"}
            else:
                # Преобразование текста в более удобный формат
                formatted_content = self._format_content_line(line, line_type)
                current_section["content"].append(formatted_content)

        if current_section["title"]:
            sections.append(current_section)

        return sections

    def _format_content_line(self, line: str, line_type: str) -> Dict[str, Any]:
        """Форматирование строки контента."""
        return {
            "text": line.strip(),
            "type": line_type,
            "formatting": {
                "is_bold": any(
                    marker in line.lower()
                    for marker in ["важно", "ключевой", "основной"]
                ),
                "is_highlight": any(
                    marker in line.lower()
                    for marker in ["пример", "заметь", "обрати внимание"]
                ),
                "color": self._determine_content_color(line, line_type),
            },
            "visual_aids": self._suggest_visual_aids(line),
        }

    async def _generate_infographics(
        self, content: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Генерация инфографики для ключевых концепций."""
        infographics = []

        # Создание диаграмм для важных концепций
        for concept in content["concepts"]:
            if concept["importance"] == "high":
                infographic = {
                    "title": concept["content"],
                    "type": "concept_diagram",
                    "elements": self._create_concept_visualization(concept),
                    "style": {
                        "color_scheme": "adhd_friendly",
                        "layout": "hierarchical",
                    },
                }
                infographics.append(infographic)

        # Создание временной шкалы для последовательных процессов
        if any(
            "шаг" in point.lower() or "этап" in point.lower()
            for point in content["key_points"]
        ):
            timeline = {
                "type": "timeline",
                "steps": self._extract_sequential_steps(content["key_points"]),
                "style": {"color_scheme": "sequential", "layout": "horizontal"},
            }
            infographics.append(timeline)

        return infographics

    def _create_mind_maps(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Создание ментальных карт."""
        # Создание графа для ментальной карты
        graph = nx.Graph()

        # Добавление центрального узла
        central_topic = content["title"]
        graph.add_node(central_topic, type="central")

        # Добавление основных концепций
        for concept in content["concepts"]:
            graph.add_node(concept["content"], type="concept")
            graph.add_edge(central_topic, concept["content"])

            # Добавление связанных элементов
            related = self._find_related_elements(concept["content"], content)
            for rel in related:
                graph.add_node(rel["content"], type="related")
                graph.add_edge(concept["content"], rel["content"])

        return {
            "nodes": list(graph.nodes(data=True)),
            "edges": list(graph.edges()),
            "layout": "radial",
            "style": {
                "central_node_color": "#FF6B6B",
                "concept_node_color": "#4ECDC4",
                "related_node_color": "#95A5A6",
            },
        }

    def _create_pomodoro_schedule(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Создание расписания Pomodoro на основе контента."""
        # Оценка сложности и объема материала
        content_volume = len(content["content"].split())
        difficulty_multiplier = {"low": 0.8, "medium": 1.0, "high": 1.2}

        # Расчет необходимого количества сессий
        estimated_sessions = math.ceil(
            (content_volume / 500) * difficulty_multiplier[content["difficulty"]]
        )

        schedule = []
        for session in range(estimated_sessions):
            is_long_break = (session + 1) % self.pomodoro_settings[
                "sessions_before_long_break"
            ] == 0

            schedule.append(
                {
                    "session_number": session + 1,
                    "work_duration": self.pomodoro_settings["work_duration"],
                    "break_duration": (
                        self.pomodoro_settings["long_break_duration"]
                        if is_long_break
                        else self.pomodoro_settings["break_duration"]
                    ),
                    "focus_points": self._get_session_focus_points(content, session),
                    "mini_goals": self._create_mini_goals(content, session),
                    "rewards": self._generate_rewards(session, is_long_break),
                }
            )

        return {
            "total_sessions": estimated_sessions,
            "schedule": schedule,
            "estimated_completion_time": self._calculate_completion_time(schedule),
            "gamification": {
                "points_per_session": 100,
                "bonus_points": self._calculate_bonus_points(schedule),
                "achievements": self._generate_achievements(estimated_sessions),
            },
        }

    async def _generate_interactive_elements(
        self, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Создание интерактивных элементов обучения."""
        return {
            "quizzes": self._generate_quizzes(content),
            "flashcards": self._create_flashcards(content),
            "progress_tracking": {
                "checkpoints": self._create_checkpoints(content),
                "achievements": self._define_achievements(content),
                "progress_indicators": self._create_progress_indicators(content),
            },
            "exercises": self._generate_exercises(content),
            "mini_games": self._create_mini_games(content),
        }

    async def _create_multisensory_elements(
        self, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Создание мультисенсорных элементов обучения."""
        return {
            "visual": {
                "diagrams": self._create_diagrams(content),
                "animations": await self._generate_animations(content),
                "color_schemes": self._generate_color_schemes(content),
            },
            "auditory": {
                "key_points_audio": self._create_audio_summaries(content),
                "sound_cues": self._generate_sound_cues(content),
            },
            "kinesthetic": {
                "interactive_exercises": self._create_interactive_exercises(content),
                "physical_activities": self._suggest_physical_activities(content),
            },
        }

    def _generate_quizzes(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерация викторин на основе контента."""
        quizzes = []

        # Создание вопросов на основе ключевых концепций
        for concept in content["concepts"]:
            if concept["importance"] == "high":
                quiz = {
                    "question": self._create_question_from_concept(concept),
                    "options": self._generate_options(concept, content),
                    "correct_answer": concept["content"],
                    "explanation": self._create_explanation(concept, content),
                    "difficulty": concept.get("difficulty", "medium"),
                    "points": self._calculate_question_points(concept),
                }
                quizzes.append(quiz)

        # Создание вопросов на основе ключевых моментов
        for point in content["key_points"]:
            quiz = {
                "question": self._create_question_from_key_point(point),
                "options": self._generate_options_from_context(point, content),
                "correct_answer": self._extract_answer_from_key_point(point),
                "explanation": self._create_explanation_from_context(point, content),
                "difficulty": "medium",
                "points": 10,
            }
            quizzes.append(quiz)

        return quizzes

    def _create_flashcards(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Создание флэш-карточек."""
        flashcards = []

        # Создание карточек из концепций
        for concept in content["concepts"]:
            card = {
                "front": concept["content"],
                "back": self._create_concept_explanation(concept, content),
                "hints": self._generate_hints(concept),
                "difficulty": concept.get("difficulty", "medium"),
                "tags": self._generate_concept_tags(concept),
                "visual_aid": self._create_visual_aid(concept),
            }
            flashcards.append(card)

        # Создание карточек из ключевых моментов
        for point in content["key_points"]:
            card = {
                "front": self._create_question_from_key_point(point),
                "back": point,
                "hints": self._generate_hints_from_context(point, content),
                "difficulty": "medium",
                "tags": self._generate_key_point_tags(point),
                "visual_aid": self._create_visual_aid_from_key_point(point),
            }
            flashcards.append(card)

        return flashcards

    async def _process_text(self, content: str, title: str = None) -> Dict[str, Any]:
        """Обработка текстового контента."""
        # Анализ текста
        words = content.split()
        sentences = [s.strip() for s in content.split(".") if s.strip()]

        # Извлечение ключевых концепций
        concepts = self._extract_concepts(content)

        # Оценка сложности
        difficulty = self._assess_difficulty(content, len(concepts))

        return {
            "content": content,
            "title": title or self._extract_title(content),
            "summary": self._generate_summary(content),
            "key_points": self._extract_key_points(content),
            "concepts": concepts,
            "difficulty": difficulty,
            "metadata": {
                "word_count": len(words),
                "sentence_count": len(sentences),
                "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            },
        }

    async def _process_audio(self, file_path: str, title: str = None) -> Dict[str, Any]:
        """Обработка аудио с помощью Whisper."""
        try:
            # Транскрибация аудио
            result = self.whisper_model.transcribe(file_path)

            # Обработка транскрибированного текста
            text_content = await self._process_text(result["text"], title)

            return {
                **text_content,
                "metadata": {
                    **text_content["metadata"],
                    "duration": result.get("duration", 0),
                    "language": result.get("language", "unknown"),
                    "original_file": file_path,
                },
            }
        except Exception as e:
            print(f"Ошибка при обработке аудио: {str(e)}")
            raise

    async def _process_video(self, file_path: str, title: str = None) -> Dict[str, Any]:
        """Обработка видео."""
        try:
            # Извлечение аудио и его обработка
            audio_result = await self._extract_and_process_audio(file_path)

            # Анализ кадров
            frames_analysis = await self._analyze_video_frames(file_path)

            return {
                **audio_result,
                "metadata": {
                    **audio_result["metadata"],
                    "frame_analysis": frames_analysis,
                    "original_file": file_path,
                },
            }
        except Exception as e:
            print(f"Ошибка при обработке видео: {str(e)}")
            raise

    async def _process_image(self, file_path: str, title: str = None) -> Dict[str, Any]:
        """Обработка изображений с OCR."""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)

            # Обработка извлеченного текста
            text_content = await self._process_text(text, title)

            return {
                **text_content,
                "metadata": {
                    **text_content["metadata"],
                    "image_size": image.size,
                    "image_mode": image.mode,
                    "original_file": file_path,
                },
            }
        except Exception as e:
            print(f"Ошибка при обработке изображения: {str(e)}")
            raise

    async def _process_pdf(self, file_path: str, title: str = None) -> Dict[str, Any]:
        """Обработка PDF документов."""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()

            # Обработка извлеченного текста
            text_content = await self._process_text(text, title)

            return {
                **text_content,
                "metadata": {
                    **text_content["metadata"],
                    "page_count": len(doc),
                    "pdf_info": doc.metadata,
                    "original_file": file_path,
                },
            }
        except Exception as e:
            print(f"Ошибка при обработке PDF: {str(e)}")
            raise

    async def _process_youtube(self, url: str, title: str = None) -> Dict[str, Any]:
        """Обработка YouTube видео."""
        try:
            import yt_dlp
            import tempfile
            import os

            # Настройка yt-dlp
            ydl_opts = {
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "extract_flat": True,
                "quiet": True,
            }

            # Получение информации о видео
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                video_info = {
                    "title": info.get("title", ""),
                    "author": info.get("uploader", ""),
                    "length": info.get("duration", 0),
                    "views": info.get("view_count", 0),
                    "description": info.get("description", ""),
                    "keywords": info.get("tags", []),
                    "thumbnail_url": info.get("thumbnail", ""),
                }

            # Создаем временную директорию для файлов
            with tempfile.TemporaryDirectory() as temp_dir:
                # Настройка для загрузки
                ydl_opts.update(
                    {
                        "outtmpl": os.path.join(temp_dir, "%(title)s.%(ext)s"),
                        "quiet": False,
                    }
                )

                # Загружаем видео
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    video_path = ydl.prepare_filename(info)

                    # Транскрибация аудио
                    audio_result = await self._process_audio(video_path)

                    # Анализ видео
                    frames_analysis = await self._analyze_video_frames(video_path)

                    # Создание структурированного контента
                    content = {
                        **audio_result,
                        "youtube_info": video_info,
                        "metadata": {
                            **audio_result["metadata"],
                            "frame_analysis": frames_analysis,
                            "original_url": url,
                            "platform": "youtube",
                        },
                    }

                    # Создание временных меток для ключевых моментов
                    content["timestamps"] = self._generate_timestamps(content)

                    # Создание интерактивных элементов
                    content["interactive_elements"] = (
                        await self._create_video_interactive_elements(content)
                    )

                    return content

        except Exception as e:
            print(f"Ошибка при обработке YouTube видео: {str(e)}")
            raise

    def _generate_timestamps(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерация временных меток для ключевых моментов видео."""
        timestamps = []

        # Анализ ключевых моментов из транскрипции
        for i, point in enumerate(content["key_points"]):
            timestamp = {
                "time": self._estimate_timestamp(point, content),
                "title": f"Ключевой момент {i + 1}",
                "content": point,
                "type": "key_point",
            }
            timestamps.append(timestamp)

        # Добавление важных концепций
        for concept in content["concepts"]:
            if concept["importance"] == "high":
                timestamp = {
                    "time": self._estimate_timestamp(concept["content"], content),
                    "title": f"Концепция: {concept['content']}",
                    "content": concept["content"],
                    "type": "concept",
                }
                timestamps.append(timestamp)

        return sorted(timestamps, key=lambda x: x["time"])

    async def _create_video_interactive_elements(
        self, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Создание интерактивных элементов для видео."""
        return {
            "chapters": self._create_video_chapters(content),
            "quick_notes": self._generate_video_notes(content),
            "key_frames": self._extract_key_frames(content),
            "quiz_points": self._generate_video_quiz_points(content),
            "visual_summary": self._create_video_visual_summary(content),
            "learning_checkpoints": self._create_video_checkpoints(content),
        }

    def _create_video_chapters(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Создание глав видео на основе контента."""
        chapters = []
        current_time = 0

        # Группировка контента по темам
        topics = self._extract_main_topics(content)

        for topic in topics:
            chapter = {
                "title": topic["title"],
                "start_time": current_time,
                "duration": self._estimate_topic_duration(topic, content),
                "summary": self._create_chapter_summary(topic),
                "key_points": self._extract_topic_key_points(topic),
                "concepts": topic["concepts"],
            }
            chapters.append(chapter)
            current_time += chapter["duration"]

        return chapters

    def _create_video_visual_summary(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Создание визуального саммари видео."""
        return {
            "timeline": self._create_video_timeline(content),
            "concept_map": self._create_video_concept_map(content),
            "key_frames_summary": self._create_key_frames_summary(content),
            "thumbnail_storyboard": self._create_thumbnail_storyboard(content),
            "topic_hierarchy": self._create_topic_hierarchy(content),
        }

    def _extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Извлечение концепций из текста."""
        concepts = []

        # Поиск определений
        definitions = re.findall(
            r"([А-Я][^.!?]*(?:это|определяется как|называется|является)[^.!?]*[.!?])",
            text,
        )
        for definition in definitions:
            concepts.append(
                {
                    "type": "definition",
                    "content": definition.strip(),
                    "importance": "high",
                }
            )

        # Поиск ключевых терминов
        terms = re.findall(r"\b[А-Я][а-я]+(?:\s+[а-я]+)*\b", text)
        for term in terms:
            if term not in [c["content"] for c in concepts]:
                concepts.append(
                    {"type": "term", "content": term, "importance": "medium"}
                )

        return concepts

    def _assess_difficulty(self, text: str, num_concepts: int) -> str:
        """Оценка сложности текста."""
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        if num_concepts > 10 or avg_word_length > 7:
            return "high"
        elif num_concepts > 5 or avg_word_length > 5:
            return "medium"
        else:
            return "low"

    def _generate_summary(self, text: str) -> str:
        """Генерация краткого содержания."""
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        summary = []

        if sentences:
            summary.append(sentences[0])
            for sentence in sentences[1:]:
                if any(
                    marker in sentence.lower()
                    for marker in [
                        "важно",
                        "ключевой",
                        "основной",
                        "главный",
                        "следует",
                        "необходимо",
                        "итак",
                        "таким образом",
                    ]
                ):
                    summary.append(sentence)

            if len(summary) < 2 and len(sentences) > 1:
                summary.append(sentences[-1])

        return ". ".join(summary)

    def _extract_key_points(self, text: str) -> List[str]:
        """Извлечение ключевых моментов."""
        key_points = []
        patterns = [
            r"([^.!?]+(?:важно|следует|необходимо|главное)[^.!?]*[.!?])",
            r"([^.!?]+(?:это|определяется как|называется|является)[^.!?]*[.!?])",
            r"([^.!?]+(?:таким образом|следовательно|поэтому|итак)[^.!?]*[.!?])",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            key_points.extend([m.strip() for m in matches])

        return key_points

    def _extract_title(self, text: str) -> str:
        """Извлечение заголовка из текста."""
        lines = text.split("\n")
        if lines:
            first_line = lines[0].strip()
            if len(first_line) <= 100 and not first_line.endswith("."):
                return first_line

        sentences = text.split(".")
        if sentences:
            first_sentence = sentences[0].strip()
            words = first_sentence.split()
            return " ".join(words[:5]) + ("..." if len(words) > 5 else "")

        return "Untitled"

    def _create_content_modules(self, content: Dict[str, Any]) -> List[ContentModule]:
        """Создание учебных модулей из контента."""
        modules = []

        # Анализ структуры контента
        topics = self._extract_main_topics(content)

        for topic in topics:
            # Определение подтем
            subtopics = self._extract_subtopics_for_topic(topic, content)

            # Определение сложности
            difficulty = self._calculate_topic_difficulty(topic, content)

            # Определение предварительных требований
            prerequisites = self._identify_prerequisites(topic, content)

            # Создание целей обучения
            objectives = self._create_learning_objectives(topic, subtopics)

            # Оценка времени
            estimated_time = self._estimate_learning_time(topic, difficulty)

            # Создание визуальных подсказок
            visual_aids = self._create_visual_aids_for_topic(topic, content)

            # Создание упрощенной версии
            simplified = self._create_simplified_version(topic, content)

            # Генерация тегов
            tags = self._generate_topic_tags(topic, content)

            module = ContentModule(
                title=topic["title"],
                subtopics=subtopics,
                difficulty_level=difficulty,
                prerequisites=prerequisites,
                learning_objectives=objectives,
                estimated_time=estimated_time,
                tags=tags,
                visual_aids=visual_aids,
                simplified_version=simplified,
            )
            modules.append(module)

        return modules

    def _categorize_content(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Умная категоризация контента."""
        categories = []

        # Основные категории
        main_categories = {
            "theoretical": {
                "keywords": ["теория", "концепция", "определение", "понятие"],
                "importance": "high",
                "color": "#FF6B6B",
            },
            "practical": {
                "keywords": ["практика", "пример", "задача", "упражнение"],
                "importance": "high",
                "color": "#4ECDC4",
            },
            "reference": {
                "keywords": ["ссылка", "источник", "дополнительно"],
                "importance": "medium",
                "color": "#95A5A6",
            },
        }

        # Анализ контента для каждой категории
        for cat_name, cat_info in main_categories.items():
            category_content = self._extract_category_content(
                content, cat_info["keywords"]
            )
            if category_content:
                categories.append(
                    {
                        "name": cat_name,
                        "content": category_content,
                        "importance": cat_info["importance"],
                        "color": cat_info["color"],
                        "subcategories": self._create_subcategories(category_content),
                    }
                )

        # Дополнительные умные категории
        smart_categories = self._generate_smart_categories(content)
        categories.extend(smart_categories)

        return categories

    def _create_content_summary(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Создание расширенного саммари контента."""
        return {
            "brief": self._create_brief_summary(content),
            "detailed": self._create_detailed_summary(content),
            "visual": self._create_visual_summary(content),
            "key_concepts": self._extract_key_concepts(content),
            "learning_path": self._create_learning_path(content),
            "difficulty_breakdown": self._analyze_difficulty_distribution(content),
            "time_estimates": self._calculate_time_estimates(content),
            "prerequisites": self._identify_global_prerequisites(content),
            "outcomes": self._define_learning_outcomes(content),
        }

    def _create_brief_summary(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Создание краткого саммари."""
        return {
            "title": content["title"],
            "main_points": self._extract_main_points(content),
            "key_takeaways": self._identify_key_takeaways(content),
            "difficulty_level": content["difficulty"],
            "estimated_time": self._calculate_total_time(content),
            "quick_start": self._create_quick_start_guide(content),
        }

    def _create_detailed_summary(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Создание детального саммари."""
        return {
            "sections": self._analyze_content_sections(content),
            "concepts": self._analyze_concepts_relationships(content),
            "progression": self._analyze_learning_progression(content),
            "challenges": self._identify_potential_challenges(content),
            "resources": self._compile_additional_resources(content),
            "practice_items": self._extract_practice_items(content),
        }

    def _create_visual_summary(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Создание визуального саммари."""
        return {
            "mind_map": self._create_summary_mind_map(content),
            "concept_diagram": self._create_concept_diagram(content),
            "progress_chart": self._create_progress_chart(content),
            "difficulty_graph": self._create_difficulty_graph(content),
            "timeline": self._create_learning_timeline(content),
            "infographics": self._create_summary_infographics(content),
        }

    def _extract_main_topics(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Извлечение основных тем из контента."""
        topics = []

        # Анализ заголовков и подзаголовков
        headers = self._extract_headers(content["content"])

        # Анализ ключевых концепций
        concepts = content["concepts"]

        # Объединение и фильтрация тем
        for header in headers:
            topic = {
                "title": header["text"],
                "type": "header",
                "importance": header["level"],
                "content": self._extract_topic_content(header, content),
                "concepts": [
                    c for c in concepts if self._is_concept_related(c, header)
                ],
            }
            topics.append(topic)

        # Добавление важных концепций как отдельных тем
        for concept in concepts:
            if concept["importance"] == "high" and not any(
                self._is_concept_in_topic(concept, topic) for topic in topics
            ):
                topic = {
                    "title": concept["content"],
                    "type": "concept",
                    "importance": "high",
                    "content": self._extract_concept_content(concept, content),
                    "concepts": [concept],
                }
                topics.append(topic)

        return topics

    def _extract_subtopics_for_topic(
        self, topic: Dict[str, Any], content: Dict[str, Any]
    ) -> List[str]:
        """Извлечение подтем для заданной темы."""
        subtopics = []

        # Анализ структуры контента
        if topic["type"] == "header":
            # Поиск подзаголовков
            subtopics.extend(self._find_subheaders(topic, content))

        # Анализ связанных концепций
        for concept in topic["concepts"]:
            related = self._find_related_concepts(concept, content)
            subtopics.extend([r["content"] for r in related])

        # Удаление дубликатов и сортировка
        return list(set(subtopics))

    def _calculate_topic_difficulty(
        self, topic: Dict[str, Any], content: Dict[str, Any]
    ) -> float:
        """Расчет сложности темы."""
        factors = {
            "concept_count": len(topic["concepts"]),
            "text_complexity": self._analyze_text_complexity(topic["content"]),
            "prerequisites": len(self._identify_prerequisites(topic, content)),
            "interconnections": len(
                self._find_concept_interconnections(topic["concepts"])
            ),
        }

        weights = {
            "concept_count": 0.3,
            "text_complexity": 0.3,
            "prerequisites": 0.2,
            "interconnections": 0.2,
        }

        difficulty = sum(factors[key] * weights[key] for key in factors)

        return min(max(difficulty, 0.0), 1.0)  # Нормализация к диапазону [0, 1]

    def _create_learning_objectives(
        self, topic: Dict[str, Any], subtopics: List[str]
    ) -> List[str]:
        """Создание целей обучения для темы."""
        objectives = []

        # Основные цели из темы
        objectives.append(f"Понять концепцию {topic['title']}")

        # Цели из концепций
        for concept in topic["concepts"]:
            if concept["importance"] == "high":
                objectives.append(f"Освоить {concept['content']}")

        # Цели из подтем
        for subtopic in subtopics:
            objectives.append(f"Изучить {subtopic}")

        # Практические цели
        if self._has_practical_elements(topic):
            objectives.append(f"Применить {topic['title']} на практике")

        return objectives

    def _create_visual_aids_for_topic(
        self, topic: Dict[str, Any], content: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Создание визуальных подсказок для темы."""
        visual_aids = []

        # Диаграммы для концепций
        for concept in topic["concepts"]:
            if concept["importance"] == "high":
                visual_aids.append(
                    {
                        "type": "concept_diagram",
                        "title": f"Диаграмма: {concept['content']}",
                        "content": self._create_concept_visualization(concept),
                    }
                )

        # Временная шкала для процессов
        if self._has_sequential_content(topic):
            visual_aids.append(
                {
                    "type": "timeline",
                    "title": f"Процесс: {topic['title']}",
                    "content": self._create_process_timeline(topic),
                }
            )

        # Ментальная карта для связей
        visual_aids.append(
            {
                "type": "mind_map",
                "title": f"Связи: {topic['title']}",
                "content": self._create_topic_mind_map(topic),
            }
        )

        return visual_aids

    def _create_simplified_version(
        self, topic: Dict[str, Any], content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Создание упрощенной версии темы."""
        return {
            "title": f"Упрощенное объяснение: {topic['title']}",
            "main_idea": self._simplify_concept(topic["title"]),
            "key_points": self._simplify_key_points(topic["concepts"]),
            "examples": self._create_simple_examples(topic),
            "visual_aids": self._create_simplified_visuals(topic),
            "practice": self._create_basic_exercises(topic),
        }


class LearningAgent:
    """Единый AI агент для обработки и адаптации образовательного контента."""

    def __init__(
        self, pinecone_api_key: str = None, pinecone_environment: str = "us-west1-gcp"
    ):
        self.storage_dir = "learning_content"
        self.ensure_storage_exists()
        self.processed_content: List[LearningContent] = []
        self.load_content()

        # Инициализация компонентов
        self.content_processor = ContentProcessor()
        self.pinecone_initialized = False
        if pinecone_api_key:
            self._init_pinecone(pinecone_api_key, pinecone_environment)
            self.personalization = PersonalizationEngine(self.pinecone_index)
            self.scheduler = LearningScheduler(self.personalization)

    def ensure_storage_exists(self):
        """Создание необходимых директорий."""
        directories = [
            self.storage_dir,
            os.path.join(self.storage_dir, "visualizations"),
            os.path.join(self.storage_dir, "exercises"),
            os.path.join(self.storage_dir, "summaries"),
        ]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def _init_pinecone(self, api_key: str, environment: str):
        """Инициализация Pinecone."""
        try:
            pinecone.init(api_key=api_key, environment=environment)
            index_name = "learning-content"
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(name=index_name, dimension=384, metric="cosine")
            self.pinecone_index = pinecone.Index(index_name)
            self.pinecone_initialized = True
        except Exception as e:
            print(f"Ошибка при инициализации Pinecone: {str(e)}")
            self.pinecone_initialized = False

    async def get_personalized_schedule(self, user_id: str) -> Dict[str, Any]:
        """Получение персонализированного расписания."""
        if not self.pinecone_initialized:
            return {}
        return await self.scheduler.create_personalized_schedule(user_id)

    def update_schedule_preferences(
        self, user_id: str, preferences: Dict[str, Any]
    ) -> None:
        """Обновление предпочтений расписания."""
        if self.pinecone_initialized:
            self.scheduler.update_preferences(user_id, preferences)

    async def process_new_content(
        self, content: str, content_type: str, title: str = None
    ) -> LearningContent:
        """Обработка нового контента."""
        try:
            # Обработка контента
            processed_data = await self.content_processor.process_content(
                content, content_type, title
            )

            # Создание объекта LearningContent
            learning_content = LearningContent(
                title=processed_data["title"],
                content=processed_data["content"],
                content_type=content_type,
                summary=processed_data["summary"],
                key_points=processed_data["key_points"],
                concepts=processed_data["concepts"],
                difficulty=processed_data["difficulty"],
                visualizations=[],  # Будет заполнено позже
                exercises=[],  # Будет заполнено позже
                learning_path=[],  # Будет заполнено позже
                metadata=processed_data["metadata"],
                processed_at=datetime.now(),
            )

            # Сохранение контента
            self._save_content(learning_content)
            self.processed_content.append(learning_content)

            # Сохранение в Pinecone, если доступно
            if self.pinecone_initialized:
                await self._store_in_pinecone(learning_content)

            return learning_content

        except Exception as e:
            print(f"Ошибка при обработке нового контента: {str(e)}")
            raise

    def _save_content(self, content: LearningContent):
        """Сохранение обработанного контента."""
        filename = f"{content.title.lower().replace(' ', '_')}_{content.processed_at.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.storage_dir, filename)

        content_dict = {
            "title": content.title,
            "content": content.content,
            "content_type": content.content_type,
            "summary": content.summary,
            "key_points": content.key_points,
            "concepts": content.concepts,
            "difficulty": content.difficulty,
            "visualizations": content.visualizations,
            "exercises": content.exercises,
            "learning_path": content.learning_path,
            "metadata": content.metadata,
            "processed_at": content.processed_at.isoformat(),
            "engagement_metrics": content.engagement_metrics,
            "adaptive_difficulty": content.adaptive_difficulty,
            "learning_progress": content.learning_progress,
            "modules": [m.__dict__ for m in content.modules],
            "tags": content.tags,
            "visual_code": content.visual_code,
            "smart_categories": content.smart_categories,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(content_dict, f, ensure_ascii=False, indent=2)


# Создаем единственный экземпляр агента
learning_agent = LearningAgent()
