from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import os
import json
from colorsys import hsv_to_rgb
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from .cache_manager import CacheManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ContentVisualizer:
    def __init__(self):
        self.storage_dir = "visualizations"
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        # Цветовая схема
        self.colors = {
            "background": "#F5F5F5",
            "primary": "#2196F3",  # Синий
            "secondary": "#4CAF50",  # Зеленый
            "accent": "#FF9800",  # Оранжевый
            "text": "#212121",
            "text_light": "#757575",
            "border": "#E0E0E0",
        }

        # Конвертируем hex в RGB
        self.colors_rgb = {
            k: tuple(int(v[i : i + 2], 16) for i in (1, 3, 5))
            for k, v in self.colors.items()
        }

        self.cache_manager = CacheManager(cache_dir="./cache/visualizations")
        self.style = {
            "figure.figsize": (10, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2,
            "lines.markersize": 8,
            "font.family": "sans-serif",
        }
        plt.style.use("seaborn")
        for key, value in self.style.items():
            plt.rcParams[key] = value

        # Директория для шаблонов
        self.templates_dir = os.path.join(self.storage_dir, "templates")
        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir)

        logger.info("Visualizer initialized")

    def _get_text_size(self, text, font):
        """Получить размер текста"""
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def _wrap_text(self, text, font, max_width):
        """Разбить текст на строки по ширине"""
        words = text.split()
        lines = []
        current_line = []
        current_width = 0

        for word in words:
            word_width = self._get_text_size(word + " ", font)[0]
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def load_template(self, template_name):
        """Загрузить шаблон инфографики"""
        template_path = os.path.join(self.templates_dir, template_name)
        if os.path.exists(template_path):
            return Image.open(template_path)
        return None

    def create_gradient(self, size, color1, color2, direction="horizontal"):
        """Создать градиентное изображение"""
        image = Image.new("RGB", size)
        draw = ImageDraw.Draw(image)

        if direction == "horizontal":
            for x in range(size[0]):
                r = int(color1[0] + (color2[0] - color1[0]) * x / size[0])
                g = int(color1[1] + (color2[1] - color1[1]) * x / size[0])
                b = int(color1[2] + (color2[2] - color1[2]) * x / size[0])
                draw.line([(x, 0), (x, size[1])], fill=(r, g, b))
        else:
            for y in range(size[1]):
                r = int(color1[0] + (color2[0] - color1[0]) * y / size[1])
                g = int(color1[1] + (color2[1] - color1[1]) * y / size[1])
                b = int(color1[2] + (color2[2] - color1[2]) * y / size[1])
                draw.line([(0, y), (size[0], y)], fill=(r, g, b))

        return image

    def create_hierarchy_diagram(self, title, elements, template_name=None):
        """Создать иерархическую диаграмму"""
        # Загружаем шаблон или создаем новый
        if template_name:
            image = self.load_template(template_name)
            if image:
                width, height = image.size
            else:
                width, height = 2000, 1200
                image = self.create_gradient(
                    (width, height), self.colors_rgb["background"], (255, 255, 255)
                )
        else:
            width, height = 2000, 1200
            image = self.create_gradient(
                (width, height), self.colors_rgb["background"], (255, 255, 255)
            )

        draw = ImageDraw.Draw(image)

        try:
            title_font = ImageFont.truetype("arial.ttf", 48)
            node_font = ImageFont.truetype("arial.ttf", 24)
        except:
            title_font = ImageFont.load_default()
            node_font = ImageFont.load_default()

        # Добавляем заголовок
        draw.text(
            (width // 2, 50),
            title,
            font=title_font,
            fill=self.colors_rgb["text"],
            anchor="mm",
        )

        # Рассчитываем позиции элементов
        level_positions = {}
        for element in elements:
            level = element["уровень"]
            if level not in level_positions:
                level_positions[level] = []
            level_positions[level].append(element)

        # Рисуем элементы уровень за уровнем
        for level in sorted(level_positions.keys()):
            elements_at_level = level_positions[level]
            y = 150 + (level - 1) * 250
            spacing = width // (len(elements_at_level) + 1)

            for i, element in enumerate(elements_at_level):
                x = spacing * (i + 1)

                # Разбиваем текст на строки
                text_lines = self._wrap_text(element["текст"], node_font, 200)

                # Рассчитываем размер блока
                line_height = 30
                box_width = 250
                box_height = len(text_lines) * line_height + 40

                # Создаем градиент для блока
                box_gradient = self.create_gradient(
                    (box_width, box_height), self.colors_rgb["primary"], (41, 128, 185)
                )

                # Рисуем узел с градиентом
                image.paste(box_gradient, (x - box_width // 2, y - box_height // 2))

                # Добавляем тень
                shadow_offset = 5
                draw.rectangle(
                    [
                        x - box_width // 2 + shadow_offset,
                        y - box_height // 2 + shadow_offset,
                        x + box_width // 2 + shadow_offset,
                        y + box_height // 2 + shadow_offset,
                    ],
                    fill=self.colors_rgb["text_light"],
                )

                # Добавляем текст построчно
                for j, line in enumerate(text_lines):
                    line_y = (
                        y - (len(text_lines) - 1) * line_height // 2 + j * line_height
                    )
                    draw.text(
                        (x, line_y), line, font=node_font, fill="white", anchor="mm"
                    )

                # Рисуем соединение с родительским элементом
                if level > 1:
                    parent_x = spacing * (i + 1)
                    parent_y = y - 250

                    # Создаем градиентную линию
                    line_gradient = self.create_gradient(
                        (2, int(np.sqrt((x - parent_x) ** 2 + (y - parent_y) ** 2))),
                        self.colors_rgb["accent"],
                        (255, 152, 0),
                    )

                    # Рисуем линию
                    draw.line(
                        [x, y - box_height // 2, parent_x, parent_y + box_height // 2],
                        fill=self.colors_rgb["accent"],
                        width=2,
                    )

        # Добавляем легенду
        legend_y = height - 100
        legend_items = [
            ("Основные категории", self.colors_rgb["primary"]),
            ("Подкатегории", self.colors_rgb["secondary"]),
            ("Связи", self.colors_rgb["accent"]),
        ]

        for i, (text, color) in enumerate(legend_items):
            x = 150 + i * 300
            draw.rectangle([x - 20, legend_y - 20, x + 20, legend_y + 20], fill=color)
            draw.text(
                (x + 40, legend_y), text, font=node_font, fill=self.colors_rgb["text"]
            )

        # Сохраняем диаграмму
        output_path = os.path.join(
            self.storage_dir, f'{title.lower().replace(" ", "_")}.png'
        )
        image.save(output_path, quality=95)
        return output_path

    def create_infographic(self, title, subtitle, elements, template_name=None):
        """Создать инфографику"""
        # Загружаем шаблон или создаем новый
        if template_name:
            image = self.load_template(template_name)
            if image:
                width, height = image.size
            else:
                width, height = 1600, 1000
                image = self.create_gradient(
                    (width, height), self.colors_rgb["background"], (255, 255, 255)
                )
        else:
            width, height = 1600, 1000
            image = self.create_gradient(
                (width, height), self.colors_rgb["background"], (255, 255, 255)
            )

        draw = ImageDraw.Draw(image)

        try:
            title_font = ImageFont.truetype("arial.ttf", 48)
            text_font = ImageFont.truetype("arial.ttf", 24)
        except:
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()

        # Добавляем заголовок
        draw.text(
            (width // 2, 50),
            title,
            font=title_font,
            fill=self.colors_rgb["text"],
            anchor="mm",
        )

        # Добавляем подзаголовок
        draw.text(
            (width // 2, 120),
            subtitle,
            font=text_font,
            fill=self.colors_rgb["text_light"],
            anchor="mm",
        )

        # Добавляем иконки и текст
        icon_size = 200
        spacing = width // (len(elements) + 1)

        for i, element in enumerate(elements):
            x = spacing * (i + 1)
            y = height // 2

            # Разбиваем текст на строки
            text_lines = self._wrap_text(element["текст"], text_font, 180)

            # Создаем градиент для иконки
            icon_gradient = self.create_gradient(
                (icon_size, icon_size), self.colors_rgb["secondary"], (46, 204, 113)
            )

            # Рисуем круг с градиентом
            draw.ellipse(
                [
                    x - icon_size // 2,
                    y - icon_size // 2,
                    x + icon_size // 2,
                    y + icon_size // 2,
                ],
                fill=self.colors_rgb["secondary"],
            )

            # Добавляем тень
            shadow_offset = 5
            draw.ellipse(
                [
                    x - icon_size // 2 + shadow_offset,
                    y - icon_size // 2 + shadow_offset,
                    x + icon_size // 2 + shadow_offset,
                    y + icon_size // 2 + shadow_offset,
                ],
                fill=self.colors_rgb["text_light"],
            )

            # Добавляем текст построчно
            text_y = y + icon_size // 2 + 40
            for j, line in enumerate(text_lines):
                draw.text(
                    (x, text_y + j * 30),
                    line,
                    font=text_font,
                    fill=self.colors_rgb["text"],
                    anchor="mm",
                )

            # Добавляем декоративные линии
            draw.line(
                [x, y + icon_size // 2 + 10, x, y + icon_size // 2 + 20],
                fill=self.colors_rgb["accent"],
                width=3,
            )

        # Добавляем легенду
        legend_y = height - 100
        legend_items = [
            ("Основные элементы", self.colors_rgb["secondary"]),
            ("Связи", self.colors_rgb["accent"]),
        ]

        for i, (text, color) in enumerate(legend_items):
            x = 150 + i * 300
            draw.rectangle([x - 20, legend_y - 20, x + 20, legend_y + 20], fill=color)
            draw.text(
                (x + 40, legend_y), text, font=text_font, fill=self.colors_rgb["text"]
            )

        # Сохраняем инфографику
        output_path = os.path.join(
            self.storage_dir, f'{title.lower().replace(" ", "_")}.png'
        )
        image.save(output_path, quality=95)
        return output_path

    def create_visualizations(
        self, visualizations_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Create all visualizations from the data"""
        results = {}

        if "типы_визуализаций" not in visualizations_data:
            return results

        for viz in visualizations_data["типы_визуализаций"]:
            viz_type = viz.get("тип", "")
            viz_name = viz.get("название", "")

            if viz_type == "схема":
                file_path = self._create_schema(viz)
                if file_path:
                    results[viz_name] = file_path
            elif viz_type == "инфографика":
                results[viz_name] = self.create_infographic(
                    viz_name,
                    viz.get("подзаголовок", ""),
                    viz["элементы"],
                    viz.get("шаблон"),
                )

        return results

    def _create_schema(self, schema_data: Dict[str, Any]) -> str:
        """Create a schema visualization"""
        try:
            # В реальной реализации здесь был бы код для создания визуализации
            # Сейчас просто создаем текстовый файл с описанием
            name = schema_data.get("название", "schema").lower().replace(" ", "_")
            file_path = os.path.join(self.storage_dir, f"{name}.txt")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Схема: {schema_data['название']}\n\n")

                # Записываем элементы схемы
                for element in schema_data.get("элементы", []):
                    indent = "  " * (element.get("уровень", 1) - 1)
                    f.write(f"{indent}• {element.get('текст', '')}\n")

            return file_path

        except Exception as e:
            print(f"Ошибка при создании схемы: {str(e)}")
            return ""

    def plot_learning_progress(
        self, progress_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Визуализация прогресса обучения

        Args:
            progress_data: Данные о прогрессе
            save_path: Путь для сохранения визуализации

        Returns:
            Optional[str]: Путь к сохраненной визуализации
        """
        try:
            # Проверка кэша
            cache_key = self.cache_manager.generate_key(
                str(progress_data), prefix="progress_plot"
            )
            cached_plot = self.cache_manager.get(cache_key)
            if cached_plot and save_path:
                return cached_plot

            # Подготовка данных
            dates = [datetime.fromisoformat(d) for d in progress_data.get("dates", [])]
            scores = progress_data.get("scores", [])
            completed_items = progress_data.get("completed_items", [])

            if not dates or not scores:
                raise ValueError("Insufficient data for visualization")

            # Создание графика
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # График оценок
            ax1.plot(dates, scores, "b-", label="Scores")
            ax1.set_title("Learning Progress Over Time")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Score")
            ax1.grid(True)
            ax1.legend()

            # График выполненных заданий
            ax2.bar(
                dates, completed_items, color="g", alpha=0.6, label="Completed Items"
            )
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Number of Completed Items")
            ax2.grid(True)
            ax2.legend()

            plt.tight_layout()

            # Сохранение
            if save_path:
                plt.savefig(save_path)
                self.cache_manager.set(cache_key, save_path)
                return save_path

            plt.close()
            return None

        except Exception as e:
            logging.error(f"Error plotting learning progress: {str(e)}")
            raise

    def plot_focus_statistics(
        self, focus_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Визуализация статистики фокусировки

        Args:
            focus_data: Данные о фокусировке
            save_path: Путь для сохранения визуализации

        Returns:
            Optional[str]: Путь к сохраненной визуализации
        """
        try:
            cache_key = self.cache_manager.generate_key(
                str(focus_data), prefix="focus_plot"
            )
            cached_plot = self.cache_manager.get(cache_key)
            if cached_plot and save_path:
                return cached_plot

            # Подготовка данных
            dates = [datetime.fromisoformat(d) for d in focus_data.get("dates", [])]
            focus_scores = focus_data.get("focus_scores", [])
            interruptions = focus_data.get("interruptions", [])

            if not dates or not focus_scores:
                raise ValueError("Insufficient data for visualization")

            # Создание графика
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # График фокус-скоров
            ax1.plot(dates, focus_scores, "r-", label="Focus Score")
            ax1.set_title("Focus Statistics Over Time")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Focus Score")
            ax1.grid(True)
            ax1.legend()

            # График прерываний
            ax2.bar(
                dates, interruptions, color="orange", alpha=0.6, label="Interruptions"
            )
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Number of Interruptions")
            ax2.grid(True)
            ax2.legend()

            plt.tight_layout()

            # Сохранение
            if save_path:
                plt.savefig(save_path)
                self.cache_manager.set(cache_key, save_path)
                return save_path

            plt.close()
            return None

        except Exception as e:
            logging.error(f"Error plotting focus statistics: {str(e)}")
            raise

    def create_topic_network(
        self, topics_data: Dict[str, List[str]], save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Создание сетевой визуализации связей между темами

        Args:
            topics_data: Данные о связях между темами
            save_path: Путь для сохранения визуализации

        Returns:
            Optional[str]: Путь к сохраненной визуализации
        """
        try:
            import networkx as nx

            cache_key = self.cache_manager.generate_key(
                str(topics_data), prefix="network_plot"
            )
            cached_plot = self.cache_manager.get(cache_key)
            if cached_plot and save_path:
                return cached_plot

            # Создание графа
            G = nx.Graph()

            # Добавление узлов и связей
            for topic, related in topics_data.items():
                G.add_node(topic)
                for related_topic in related:
                    G.add_edge(topic, related_topic)

            # Визуализация
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)

            # Отрисовка узлов
            nx.draw_networkx_nodes(
                G, pos, node_color="lightblue", node_size=1000, alpha=0.6
            )
            nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5)
            nx.draw_networkx_labels(G, pos, font_size=8)

            plt.title("Topic Relationships Network")
            plt.axis("off")

            # Сохранение
            if save_path:
                plt.savefig(save_path)
                self.cache_manager.set(cache_key, save_path)
                return save_path

            plt.close()
            return None

        except Exception as e:
            logging.error(f"Error creating topic network: {str(e)}")
            raise

    def plot_quiz_results(
        self, quiz_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Визуализация результатов тестирования

        Args:
            quiz_data: Данные о результатах тестов
            save_path: Путь для сохранения визуализации

        Returns:
            Optional[str]: Путь к сохраненной визуализации
        """
        try:
            cache_key = self.cache_manager.generate_key(
                str(quiz_data), prefix="quiz_plot"
            )
            cached_plot = self.cache_manager.get(cache_key)
            if cached_plot and save_path:
                return cached_plot

            # Подготовка данных
            scores = quiz_data.get("scores", [])
            categories = quiz_data.get("categories", [])
            correct_answers = quiz_data.get("correct_answers", [])
            total_questions = quiz_data.get("total_questions", [])

            if not scores or not categories:
                raise ValueError("Insufficient data for visualization")

            # Создание графика
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # График оценок по категориям
            ax1.bar(categories, scores, color="purple", alpha=0.6)
            ax1.set_title("Quiz Scores by Category")
            ax1.set_xlabel("Category")
            ax1.set_ylabel("Score (%)")
            ax1.grid(True)
            plt.xticks(rotation=45)

            # График правильных ответов
            accuracy = [c / t * 100 for c, t in zip(correct_answers, total_questions)]
            ax2.bar(categories, accuracy, color="green", alpha=0.6)
            ax2.set_title("Answer Accuracy by Category")
            ax2.set_xlabel("Category")
            ax2.set_ylabel("Accuracy (%)")
            ax2.grid(True)
            plt.xticks(rotation=45)

            plt.tight_layout()

            # Сохранение
            if save_path:
                plt.savefig(save_path)
                self.cache_manager.set(cache_key, save_path)
                return save_path

            plt.close()
            return None

        except Exception as e:
            logging.error(f"Error plotting quiz results: {str(e)}")
            raise

    def create_heatmap(
        self,
        data: np.ndarray,
        labels: Dict[str, List[str]],
        title: str,
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Создание тепловой карты

        Args:
            data: Данные для визуализации
            labels: Метки для осей
            title: Заголовок визуализации
            save_path: Путь для сохранения визуализации

        Returns:
            Optional[str]: Путь к сохраненной визуализации
        """
        try:
            cache_key = self.cache_manager.generate_key(
                f"{str(data)}_{str(labels)}_{title}", prefix="heatmap"
            )
            cached_plot = self.cache_manager.get(cache_key)
            if cached_plot and save_path:
                return cached_plot

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                data,
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                xticklabels=labels.get("x", []),
                yticklabels=labels.get("y", []),
            )

            plt.title(title)
            plt.tight_layout()

            # Сохранение
            if save_path:
                plt.savefig(save_path)
                self.cache_manager.set(cache_key, save_path)
                return save_path

            plt.close()
            return None

        except Exception as e:
            logging.error(f"Error creating heatmap: {str(e)}")
            raise

    def plot_time_distribution(
        self, time_data: Dict[str, List[float]], save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Визуализация распределения времени

        Args:
            time_data: Данные о распределении времени
            save_path: Путь для сохранения визуализации

        Returns:
            Optional[str]: Путь к сохраненной визуализации
        """
        try:
            cache_key = self.cache_manager.generate_key(
                str(time_data), prefix="time_plot"
            )
            cached_plot = self.cache_manager.get(cache_key)
            if cached_plot and save_path:
                return cached_plot

            # Подготовка данных
            activities = list(time_data.keys())
            times = list(time_data.values())

            if not activities or not times:
                raise ValueError("Insufficient data for visualization")

            # Создание графика
            plt.figure(figsize=(10, 6))
            plt.pie(times, labels=activities, autopct="%1.1f%%", startangle=90)

            plt.title("Time Distribution by Activity")
            plt.axis("equal")

            # Сохранение
            if save_path:
                plt.savefig(save_path)
                self.cache_manager.set(cache_key, save_path)
                return save_path

            plt.close()
            return None

        except Exception as e:
            logging.error(f"Error plotting time distribution: {str(e)}")
            raise

    def generate_mindmap(self, topics, highlighted_terms, save_path=None):
        """
        Генерирует интеллект-карту на основе тем и ключевых терминов

        Args:
            topics: Словарь тем и подтем
            highlighted_terms: Словарь выделенных терминов
            save_path: Путь для сохранения файла

        Returns:
            str: Путь к сохраненному файлу
        """
        try:
            import networkx as nx

            # Создаем граф
            G = nx.Graph()

            # Добавляем узлы и связи
            for topic, subtopics in topics.items():
                G.add_node(topic, type="topic")
                for subtopic in subtopics:
                    G.add_node(subtopic, type="subtopic")
                    G.add_edge(topic, subtopic)

            # Добавляем выделенные термины
            for term, color in highlighted_terms.items():
                G.add_node(term, type="term", color=color)
                # Связываем с наиболее релевантной темой
                for topic in topics:
                    if term.lower() in topic.lower():
                        G.add_edge(topic, term)
                        break

            # Визуализация
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G, k=0.5, iterations=50)

            # Рисуем узлы разных типов
            topic_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "topic"]
            subtopic_nodes = [
                n for n, d in G.nodes(data=True) if d.get("type") == "subtopic"
            ]
            term_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "term"]

            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=topic_nodes,
                node_color="lightblue",
                node_size=1000,
                alpha=0.8,
            )
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=subtopic_nodes,
                node_color="lightgreen",
                node_size=700,
                alpha=0.6,
            )
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=term_nodes,
                node_color="orange",
                node_size=500,
                alpha=0.7,
            )

            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
            nx.draw_networkx_labels(G, pos, font_size=8)

            plt.title("Mind Map")
            plt.axis("off")

            # Сохранение
            if save_path:
                plt.savefig(save_path)
                return save_path

            # Если путь не указан, сохраняем во временный файл
            temp_path = os.path.join(self.storage_dir, "mindmap.png")
            plt.savefig(temp_path)
            plt.close()
            return temp_path

        except Exception as e:
            logging.error(f"Error generating mindmap: {str(e)}")
            return ""

    def create_concept_hierarchy(self, topics, highlighted_terms, save_path=None):
        """
        Создает иерархию концепций

        Args:
            topics: Словарь тем и подтем
            highlighted_terms: Словарь выделенных терминов
            save_path: Путь для сохранения файла

        Returns:
            str: Путь к сохраненному файлу
        """
        try:
            # Создаем иерархическую диаграмму
            elements = []
            level = 1

            # Добавляем темы как элементы первого уровня
            for topic, subtopics in topics.items():
                elements.append({"уровень": level, "текст": topic})

                # Добавляем подтемы как элементы второго уровня
                for subtopic in subtopics:
                    elements.append({"уровень": level + 1, "текст": subtopic})

            # Создаем диаграмму
            return self.create_hierarchy_diagram(
                "Концептуальная иерархия", elements, save_path
            )

        except Exception as e:
            logging.error(f"Error creating concept hierarchy: {str(e)}")
            return ""

    def generate_learning_infographic(self, stats, save_path=None):
        """
        Создает инфографику с данными об обучении

        Args:
            stats: Статистика обучения
            save_path: Путь для сохранения файла

        Returns:
            str: Путь к сохраненному файлу
        """
        try:
            # Создаем элементы для инфографики
            elements = [
                {
                    "текст": f"Уровень завершения: {stats.get('completion_rate', 0) * 100:.1f}%"
                },
                {
                    "текст": f"Оценка вовлеченности: {stats.get('engagement_score', 0) * 100:.1f}%"
                },
                {
                    "текст": f"Уровень понимания: {stats.get('understanding_level', 0) * 100:.1f}%"
                },
            ]

            # Создаем инфографику
            return self.create_infographic(
                "Статистика обучения", "Ваш прогресс и достижения", elements, save_path
            )

        except Exception as e:
            logging.error(f"Error generating learning infographic: {str(e)}")
            return ""
