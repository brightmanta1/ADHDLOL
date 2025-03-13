import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os
import json
from datetime import datetime
from types import ModuleType

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.content_converter import ContentConverter
from utils.visualization import ContentVisualizer
from utils.content_processor import ContentProcessor
from utils.video_processor import VideoProcessor
from utils.user_behavior_analyzer import UserBehaviorAnalyzer
from utils.progress_tracker import ProgressTracker
from utils.quiz_generator import QuizGenerator
from utils.text_processor import TextProcessor
from utils.ai_models import AIModel
from utils.focus_tools import FocusTools


@pytest.fixture
def mock_ai_responses():
    """Фикстура с моками ответов AI для разных типов контента"""
    return {
        "нейробиология": {
            "category_hierarchy": """Нейробиология
- Методы исследования
  - Оптогенетика
  - Визуализация
- Результаты
  - Основные находки
  - Применение""",
            "simplified_content": """Основные моменты исследования:
1. Методология
- Использование оптогенетики
- Визуализация нейронов

2. Результаты
- Обнаружены новые механизмы
- Практическое применение""",
            "term_explanations": {
                "оптогенетика": {
                    "определение": "Метод управления нейронами с помощью света",
                    "пример": "Активация нейронов светом",
                    "связанные_термины": ["нейроны", "свет"],
                }
            },
            "visualizations": {
                "типы_визуализаций": [
                    {
                        "тип": "схема",
                        "название": "Основные концепции: Нейробиология",
                        "элементы": [
                            {"уровень": 1, "текст": "Нейробиология"},
                            {"уровень": 2, "текст": "Методы исследования"},
                            {"уровень": 3, "текст": "Оптогенетика"},
                            {"уровень": 3, "текст": "Визуализация"},
                            {"уровень": 2, "текст": "Результаты"},
                            {"уровень": 3, "текст": "Основные находки"},
                            {"уровень": 3, "текст": "Применение"},
                        ],
                    }
                ]
            },
        },
        "сдвг": {
            "category_hierarchy": """СДВГ и Нейробиология
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
            "simplified_content": """Основные аспекты СДВГ:
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
            "term_explanations": {
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
            },
            "visualizations": {
                "типы_визуализаций": [
                    {
                        "тип": "схема",
                        "название": "Основные концепции: СДВГ",
                        "элементы": [
                            {"уровень": 1, "текст": "СДВГ"},
                            {"уровень": 2, "текст": "Симптомы"},
                            {"уровень": 3, "текст": "Невнимательность"},
                            {"уровень": 3, "текст": "Гиперактивность"},
                            {"уровень": 3, "текст": "Импульсивность"},
                            {"уровень": 2, "текст": "Механизмы"},
                            {"уровень": 3, "текст": "Нейромедиаторы"},
                            {"уровень": 3, "текст": "Мозговые структуры"},
                            {"уровень": 2, "текст": "Лечение"},
                            {"уровень": 3, "текст": "Медикаментозное"},
                            {"уровень": 3, "текст": "Поведенческая терапия"},
                        ],
                    }
                ]
            },
        },
    }


@pytest.fixture
def mock_components(mock_ai_responses):
    """Фикстура для создания компонентов системы с моками"""
    mock_ai_model = AsyncMock()

    def generate_side_effect(*args, **kwargs):
        content_type = "сдвг" if "СДВГ" in args[1] else "нейробиология"
        if "категории" in args[1].lower():
            return mock_ai_responses[content_type]["category_hierarchy"]
        elif "упростить" in args[1].lower():
            return mock_ai_responses[content_type]["simplified_content"]
        elif "термины" in args[1].lower():
            return json.dumps(mock_ai_responses[content_type]["term_explanations"])
        elif "визуализации" in args[1].lower():
            return json.dumps(mock_ai_responses[content_type]["visualizations"])
        elif "тест" in args[1].lower():
            return json.dumps(
                {
                    "вопросы": [
                        {
                            "вопрос": f"Что такое {content_type.upper()}?",
                            "варианты": [
                                mock_ai_responses[content_type]["term_explanations"][
                                    (
                                        content_type.upper()
                                        if content_type == "сдвг"
                                        else "оптогенетика"
                                    )
                                ]["определение"],
                                "Неверный ответ 1",
                                "Неверный ответ 2",
                                "Неверный ответ 3",
                            ],
                            "правильный_ответ": 0,
                        }
                    ]
                }
            )
        else:
            return json.dumps(
                {
                    "рекомендации": ["Делать перерывы", "Использовать таймер"],
                    "упражнения": ["Дыхательная техника", "Разминка"],
                }
            )

    mock_ai_model.generate.side_effect = generate_side_effect

    return {
        "converter": ContentConverter(ai_model=mock_ai_model),
        "visualizer": ContentVisualizer(),
        "text_processor": TextProcessor(),
        "quiz_generator": QuizGenerator(ai_model=mock_ai_model),
        "focus_tools": FocusTools(),
        "progress_tracker": ProgressTracker(),
    }


@pytest.mark.asyncio
async def test_process_content(mock_components):
    """Тест обработки различных типов контента"""

    # Создаем основные процессоры
    content_processor = ContentProcessor(
        mock_components["converter"], mock_components["visualizer"]
    )
    video_processor = VideoProcessor(content_processor)

    # Создаем анализатор поведения пользователя
    behavior_analyzer = UserBehaviorAnalyzer(api_key="test-key", environment="test-env")

    # Тест 1: Обработка текстового контента
    text_content = """
    Исследование нейронных механизмов с помощью оптогенетики.
    Методы включают визуализацию и контроль нейронов.
    """

    processed_raw_text = mock_components["text_processor"].process_text(text_content)
    assert len(processed_raw_text.original) > 0
    assert len(processed_raw_text.simplified) > 0
    assert len(processed_raw_text.topics) > 0

    processed_text = await content_processor.process_content(
        title="Нейронные механизмы",
        content=processed_raw_text.simplified,
        categories={"нейробиология", "оптогенетика"},
    )

    # Тест 2: Обработка видео про СДВГ
    with patch("pytube.YouTube", autospec=True) as mock_youtube:
        mock_yt = mock_youtube.return_value
        mock_yt._title = None
        mock_yt._description = None
        mock_yt._length = None
        mock_yt._thumbnail_url = None
        mock_yt._keywords = None
        mock_yt.watch_url = "https://youtube.com/watch?v=_HURE27oTX4"
        mock_yt.vid_info = {
            "videoDetails": {
                "title": "СДВГ: Нейробиология и современные подходы к лечению",
                "description": """
                Подробный разбор СДВГ с точки зрения нейробиологии.
                
                00:00 Введение
                03:15 Что такое СДВГ
                07:45 Симптомы и проявления
                15:30 Нейробиологические механизмы
                25:00 Современные методы лечения
                35:00 Практические рекомендации
                
                #СДВГ #нейробиология #психология #медицина
                """,
                "lengthSeconds": "2400",
                "thumbnail": {
                    "thumbnails": [{"url": "https://example.com/adhd_thumbnail.jpg"}]
                },
                "keywords": ["СДВГ", "нейробиология", "психология", "лечение"],
            }
        }
        mock_yt.check_availability = MagicMock()

        video_content = await video_processor.process_video(
            "https://www.youtube.com/watch?v=_HURE27oTX4"
        )

        # Отслеживаем взаимодействие
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

        mock_components["progress_tracker"].update_progress(
            user_id="test_user",
            content_id=video_content.title,
            progress=0.3,
            completed_sections=["Введение"],
        )

        # Генерируем образовательные материалы
        video_quiz = await mock_components["quiz_generator"].generate_quiz(
            video_content.description
        )
        focus_recommendations = mock_components["focus_tools"].get_recommendations()

        # Проверяем результаты
        assert video_content.title == mock_yt.vid_info["videoDetails"]["title"]
        assert len(video_content.chapters) == 6
        assert "СДВГ" in video_content.categories
        assert "нейробиология" in video_content.categories
        assert len(video_content.key_points) > 0
        assert video_content.duration == 2400
        assert len(video_quiz["вопросы"]) > 0
        assert "СДВГ" in video_quiz["вопросы"][0]["вопрос"]
        assert len(focus_recommendations) > 0

        # Проверяем аналитику
        analysis = behavior_analyzer.analyze_user_behavior("test_user")
        progress = mock_components["progress_tracker"].get_progress("test_user")

        assert analysis["total_actions"] >= 1
        assert "watch_video" in analysis["action_types"]
        assert any(time > 0 for time in analysis["time_patterns"].values())
        assert len(progress) >= 1
        assert any(p["progress"] > 0 for p in progress.values())

        # Проверяем хранилище
        assert os.path.exists(content_processor.storage_dir)
        assert os.path.exists(video_processor.storage_dir)
        assert os.path.exists(behavior_analyzer.storage_dir)

        # Очищаем
        for directory in [
            content_processor.storage_dir,
            video_processor.storage_dir,
            behavior_analyzer.storage_dir,
        ]:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    os.remove(os.path.join(directory, file))
                os.rmdir(directory)

        behavior_analyzer.cleanup()
