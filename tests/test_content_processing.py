import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os
import json
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.content_converter import ContentConverter
from utils.visualization import ContentVisualizer
from utils.content_processor import ContentProcessor
from utils.video_processor import VideoProcessor
from utils.user_behavior_analyzer import UserBehaviorAnalyzer
from utils.progress_tracker import ProgressTracker
from utils.quiz_generator import QuizGenerator
from utils.text_processing import TextProcessor
from utils.ai_models import AIModel
from utils.focus_tools import FocusTools


@pytest.mark.asyncio
async def test_unified_content_processing():
    """Единый тест для проверки всех компонентов системы"""

    # Mock AI model responses
    mock_ai_model = AsyncMock()
    mock_ai_model.generate.side_effect = [
        # 1. Category hierarchy
        """Нейробиология
- Методы исследования
  - Оптогенетика
  - Визуализация
- Результаты
  - Основные находки
  - Применение""",
        # 2. Simplified content
        """Основные моменты исследования:
1. Методология
- Использование оптогенетики
- Визуализация нейронов

2. Результаты
- Обнаружены новые механизмы
- Практическое применение""",
        # 3. Term explanations
        json.dumps(
            {
                "оптогенетика": {
                    "определение": "Метод управления нейронами с помощью света",
                    "пример": "Активация нейронов светом",
                    "связанные_термины": ["нейроны", "свет"],
                }
            }
        ),
        # 4. Visualizations
        json.dumps(
            {
                "типы_визуализаций": [
                    {
                        "тип": "схема",
                        "название": "Методы исследования",
                        "элементы": [
                            {"уровень": 1, "текст": "Исследование"},
                            {"уровень": 2, "текст": "Методы"},
                        ],
                    }
                ]
            }
        ),
        # 5. Learning aids
        json.dumps(
            {
                "мнемонические_правила": [
                    {
                        "термин": "Оптогенетика",
                        "правило": "Свет + Гены = Контроль",
                        "объяснение": "Управление генетически модифицированными нейронами светом",
                    }
                ],
                "ключевые_факты": ["Метод использует свет"],
                "вопросы": ["Как работает метод?"],
            }
        ),
        # 6. Quiz questions
        json.dumps(
            {
                "вопросы": [
                    {
                        "вопрос": "Что такое оптогенетика?",
                        "варианты": [
                            "Метод управления нейронами с помощью света",
                            "Метод изучения генов",
                            "Способ визуализации мозга",
                            "Техника микроскопии",
                        ],
                        "правильный_ответ": 0,
                    }
                ]
            }
        ),
        # 7. Focus recommendations
        json.dumps(
            {
                "рекомендации": [
                    "Делать перерывы каждые 25 минут",
                    "Использовать таймер",
                    "Вести записи",
                ],
                "упражнения": ["Дыхательная техника", "Короткая разминка"],
            }
        ),
    ]

    # Create components
    converter = ContentConverter(ai_model=mock_ai_model)
    visualizer = ContentVisualizer()
    content_processor = ContentProcessor(converter, visualizer)
    video_processor = VideoProcessor(content_processor)
    text_processor = TextProcessor()
    quiz_generator = QuizGenerator(ai_model=mock_ai_model)
    focus_tools = FocusTools()
    progress_tracker = ProgressTracker()

    # Mock Pinecone
    with patch("pinecone.init"), patch("pinecone.list_indexes", return_value=[]), patch(
        "pinecone.create_index"
    ), patch("pinecone.Index") as mock_index:

        # Configure mock index
        mock_pinecone_index = MagicMock()
        mock_pinecone_index.upsert = MagicMock()
        mock_pinecone_index.query = MagicMock(return_value=MagicMock(matches=[]))
        mock_index.return_value = mock_pinecone_index

        # Create behavior analyzer
        behavior_analyzer = UserBehaviorAnalyzer(
            api_key="test-key", environment="test-env"
        )

        # Test 1: Process text content
        text_content = """
        Исследование нейронных механизмов с помощью оптогенетики.
        Методы включают визуализацию и контроль нейронов.
        """

        # Process text with TextProcessor
        processed_raw_text = text_processor.process_text(text_content)
        assert len(processed_raw_text) > 0

        # Process content with ContentProcessor
        processed_text = await content_processor.process_content(
            title="Нейронные механизмы",
            content=processed_raw_text,
            categories={"нейробиология", "оптогенетика"},
        )

        # Track text content interaction and progress
        action_id = await behavior_analyzer.track_user_action(
            user_id="test_user",
            action_type="view_text",
            content_id=processed_text.title,
            metadata={"categories": list(processed_text.categories)},
        )

        progress_tracker.update_progress(
            user_id="test_user",
            content_id=processed_text.title,
            progress=0.5,
            completed_sections=["Введение", "Методы"],
        )

        # Generate quiz
        quiz = await quiz_generator.generate_quiz(processed_text.content)
        assert len(quiz["вопросы"]) > 0

        # Get focus recommendations
        focus_recommendations = focus_tools.get_recommendations()
        assert len(focus_recommendations) > 0

        # Verify text processing
        assert processed_text.title == "Нейронные механизмы"
        assert len(processed_text.categories) == 2
        assert "оптогенетика" in processed_text.categories
        assert len(processed_text.visualization_paths) > 0

        # Test 2: Process video content
        with patch("pytube.YouTube") as mock_youtube:
            mock_yt = MagicMock()
            mock_yt.title = "Оптогенетика в исследованиях"
            mock_yt.description = """
            00:00 Введение
            02:15 Методы
            05:30 Результаты
            
            #нейробиология #оптогенетика
            """
            mock_yt.length = 600
            mock_yt.thumbnail_url = "https://example.com/thumb.jpg"
            mock_yt.keywords = ["нейробиология", "оптогенетика"]

            mock_youtube.return_value = mock_yt

            video_content = await video_processor.process_video(
                "https://www.youtube.com/watch?v=example"
            )

            # Track video content interaction and progress
            await behavior_analyzer.track_user_action(
                user_id="test_user",
                action_type="watch_video",
                content_id=video_content.title,
                metadata={
                    "duration": video_content.duration,
                    "categories": list(video_content.categories),
                },
            )

            progress_tracker.update_progress(
                user_id="test_user",
                content_id=video_content.title,
                progress=0.3,
                completed_sections=["Введение"],
            )

            # Generate quiz for video content
            video_quiz = await quiz_generator.generate_quiz(video_content.description)
            assert len(video_quiz["вопросы"]) > 0

            # Verify video processing
            assert video_content.title == mock_yt.title
            assert len(video_content.chapters) == 3
            assert "оптогенетика" in video_content.categories
            assert len(video_content.key_points) > 0

        # Test 3: Analyze user behavior and progress
        analysis = behavior_analyzer.analyze_user_behavior("test_user")
        progress = progress_tracker.get_progress("test_user")

        # Verify behavior analysis
        assert analysis["total_actions"] >= 2
        assert "view_text" in analysis["action_types"]
        assert "watch_video" in analysis["action_types"]
        assert any(time > 0 for time in analysis["time_patterns"].values())

        # Verify progress tracking
        assert len(progress) >= 2
        assert any(p["progress"] > 0 for p in progress.values())
        assert any(len(p["completed_sections"]) > 0 for p in progress.values())

        # Verify storage
        assert os.path.exists(content_processor.storage_dir)
        assert os.path.exists(video_processor.storage_dir)
        assert os.path.exists(behavior_analyzer.storage_dir)

        # Verify files were created
        text_files = os.listdir(content_processor.storage_dir)
        video_files = os.listdir(video_processor.storage_dir)
        behavior_files = os.listdir(behavior_analyzer.storage_dir)
        assert len(text_files) >= 1
        assert len(video_files) >= 1
        assert len(behavior_files) >= 2

        # Clean up
        for file in text_files:
            os.remove(os.path.join(content_processor.storage_dir, file))
        for file in video_files:
            os.remove(os.path.join(video_processor.storage_dir, file))
        for file in behavior_files:
            os.remove(os.path.join(behavior_analyzer.storage_dir, file))
        os.rmdir(content_processor.storage_dir)
        os.rmdir(video_processor.storage_dir)
        os.rmdir(behavior_analyzer.storage_dir)

        # Cleanup Pinecone
        behavior_analyzer.cleanup()


@pytest.mark.asyncio
async def test_specific_youtube_video():
    """Тест обработки конкретного YouTube видео"""

    # Mock AI model responses for specific video content
    mock_ai_model = AsyncMock()
    mock_ai_model.generate.side_effect = [
        # 1. Category hierarchy
        """СДВГ и Нейробиология
- Симптомы
  - Невнимательность
  - Гиперактивность
  - Импульсивность
- Механизмы
  - Нейромедиаторы
  - Мозговые структуры
- Лечение
  - Медикаментозное
  - Поведенческая терапия""",
        # 2. Simplified content
        """Основные аспекты СДВГ:
1. Симптоматика
- Трудности с концентрацией
- Повышенная активность
- Импульсивные решения

2. Биологические основы
- Особенности работы мозга
- Роль нейромедиаторов

3. Подходы к лечению
- Медикаменты
- Терапевтические методики""",
        # 3. Term explanations
        json.dumps(
            {
                "СДВГ": {
                    "определение": "Синдром дефицита внимания и гиперактивности",
                    "пример": "Трудности с концентрацией и усидчивостью",
                    "связанные_термины": [
                        "внимание",
                        "гиперактивность",
                        "импульсивность",
                    ],
                },
                "нейромедиаторы": {
                    "определение": "Химические вещества, передающие сигналы в мозге",
                    "пример": "Дофамин, норадреналин",
                    "связанные_термины": ["мозг", "передача сигналов"],
                },
            }
        ),
        # 4. Visualizations
        json.dumps(
            {
                "типы_визуализаций": [
                    {
                        "тип": "схема",
                        "название": "Симптомы СДВГ",
                        "элементы": [
                            {"уровень": 1, "текст": "СДВГ"},
                            {"уровень": 2, "текст": "Невнимательность"},
                            {"уровень": 2, "текст": "Гиперактивность"},
                            {"уровень": 2, "текст": "Импульсивность"},
                        ],
                    }
                ]
            }
        ),
        # 5. Learning aids
        json.dumps(
            {
                "мнемонические_правила": [
                    {
                        "термин": "СДВГ",
                        "правило": "Сложно Долго Внимание Где-то",
                        "объяснение": "Помогает запомнить основную характеристику - трудности с вниманием",
                    }
                ],
                "ключевые_факты": [
                    "СДВГ - нейробиологическое расстройство",
                    "Влияет на внимание и поведение",
                    "Требует комплексного подхода к лечению",
                ],
            }
        ),
        # 6. Quiz questions
        json.dumps(
            {
                "вопросы": [
                    {
                        "вопрос": "Какие основные симптомы СДВГ?",
                        "варианты": [
                            "Невнимательность, гиперактивность, импульсивность",
                            "Усталость, сонливость, апатия",
                            "Агрессия, тревожность, депрессия",
                            "Головная боль, тошнота, головокружение",
                        ],
                        "правильный_ответ": 0,
                    }
                ]
            }
        ),
    ]

    # Create components
    converter = ContentConverter(ai_model=mock_ai_model)
    visualizer = ContentVisualizer()
    content_processor = ContentProcessor(converter, visualizer)
    video_processor = VideoProcessor(content_processor)
    quiz_generator = QuizGenerator(ai_model=mock_ai_model)
    focus_tools = FocusTools()
    progress_tracker = ProgressTracker()

    # Mock Pinecone for behavior tracking
    with patch("pinecone.init"), patch("pinecone.list_indexes", return_value=[]), patch(
        "pinecone.create_index"
    ), patch("pinecone.Index") as mock_index:

        mock_pinecone_index = MagicMock()
        mock_pinecone_index.upsert = MagicMock()
        mock_pinecone_index.query = MagicMock(return_value=MagicMock(matches=[]))
        mock_index.return_value = mock_pinecone_index

        behavior_analyzer = UserBehaviorAnalyzer(
            api_key="test-key", environment="test-env"
        )

        # Process specific YouTube video
        with patch("pytube.YouTube") as mock_youtube:
            mock_yt = MagicMock()
            mock_yt.title = "СДВГ: Нейробиология и современные подходы к лечению"
            mock_yt.description = """
            Подробный разбор СДВГ с точки зрения нейробиологии.
            
            00:00 Введение
            03:15 Что такое СДВГ
            07:45 Симптомы и проявления
            15:30 Нейробиологические механизмы
            25:00 Современные методы лечения
            35:00 Практические рекомендации
            
            #СДВГ #нейробиология #психология #медицина
            """
            mock_yt.length = 2400  # 40 минут
            mock_yt.thumbnail_url = "https://example.com/adhd_thumbnail.jpg"
            mock_yt.keywords = ["СДВГ", "нейробиология", "психология", "лечение"]

            mock_youtube.return_value = mock_yt

            # Process video
            video_content = await video_processor.process_video(
                "https://www.youtube.com/watch?v=_HURE27oTX4"
            )

            # Track interaction
            await behavior_analyzer.track_user_action(
                user_id="test_user",
                action_type="watch_video",
                content_id=video_content.title,
                metadata={
                    "duration": video_content.duration,
                    "categories": list(video_content.categories),
                    "completion_rate": 0.0,
                },
            )

            # Generate educational materials
            video_quiz = await quiz_generator.generate_quiz(video_content.description)
            focus_recommendations = focus_tools.get_recommendations()

            # Verify video processing
            assert video_content.title == mock_yt.title
            assert len(video_content.chapters) == 6  # Проверяем количество глав
            assert "СДВГ" in video_content.categories
            assert "нейробиология" in video_content.categories
            assert len(video_content.key_points) > 0
            assert video_content.duration == 2400

            # Verify quiz generation
            assert len(video_quiz["вопросы"]) > 0
            assert "СДВГ" in video_quiz["вопросы"][0]["вопрос"]

            # Verify focus tools
            assert len(focus_recommendations) > 0

            # Clean up
            behavior_analyzer.cleanup()
