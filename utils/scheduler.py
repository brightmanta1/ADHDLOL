"""
Модуль планирования обучения для ADHDLearningCompanion
"""

from datetime import datetime
from typing import Dict, List, Any
import plotly.graph_objects as go
from .personalization import PersonalizationEngine


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

            # Добавление основных сессий
            fig.add_trace(
                go.Bar(
                    x=times,
                    y=durations,
                    name="Учебные сессии",
                    marker_color="rgb(55, 83, 109)",
                )
            )

            # Добавление перерывов
            fig.add_trace(
                go.Bar(
                    x=times, y=breaks, name="Перерывы", marker_color="rgb(26, 118, 255)"
                )
            )

            fig.update_layout(
                title="Расписание занятий",
                xaxis_title="Время",
                yaxis_title="Длительность (минуты)",
                barmode="group",
            )

            return fig
        except Exception as e:
            print(f"Ошибка при создании визуализации: {str(e)}")
            return None

    def _recommend_activity_type(self, focus_score: float) -> str:
        """Рекомендация типа активности на основе оценки фокусировки."""
        if focus_score > 0.8:
            return "Сложные задачи"
        elif focus_score > 0.6:
            return "Средние задачи"
        else:
            return "Легкие задачи"

    def _calculate_schedule_effectiveness(
        self, user_id: str, schedule: Dict[str, Any]
    ) -> float:
        """Расчет эффективности расписания."""
        try:
            # Базовая эффективность
            base_effectiveness = 0.7

            # Учет оптимальных времен
            time_alignment = self._calculate_time_alignment(schedule)

            # Учет предпочтений пользователя
            preference_alignment = self._calculate_preference_alignment(
                user_id, schedule
            )

            # Взвешенная сумма факторов
            effectiveness = (
                base_effectiveness * 0.4
                + time_alignment * 0.3
                + preference_alignment * 0.3
            )

            return round(effectiveness, 2)
        except Exception as e:
            print(f"Ошибка при расчете эффективности: {str(e)}")
            return 0.5

    def _calculate_time_alignment(self, schedule: Dict[str, Any]) -> float:
        """Расчет соответствия оптимальным временам."""
        return 0.8  # Упрощенная реализация

    def _calculate_preference_alignment(
        self, user_id: str, schedule: Dict[str, Any]
    ) -> float:
        """Расчет соответствия предпочтениям пользователя."""
        return 0.75  # Упрощенная реализация
