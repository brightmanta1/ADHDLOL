"""
Модуль персонализации для ADHDLearningCompanion
"""

import json
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from .obfuscator import CodeObfuscator


@CodeObfuscator.protect_class
class PersonalizationEngine:
    """Движок персонализации на основе Pinecone."""

    @CodeObfuscator.obfuscate_function
    def __init__(self, pinecone_index):
        self.index = pinecone_index
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.namespace_users = CodeObfuscator.obfuscate_string("user_profiles")
        self.namespace_patterns = CodeObfuscator.obfuscate_string("learning_patterns")
        self.namespace_content = CodeObfuscator.obfuscate_string("learning_content")

    @CodeObfuscator.obfuscate_function
    async def update_user_profile(self, user_id: str, interaction_data: Dict[str, Any]):
        """Обновление профиля пользователя."""
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

    def _time_to_minutes(self, t: str) -> int:
        """Конвертация времени в минуты."""
        hours, minutes = map(int, t.split(":"))
        return hours * 60 + minutes

    def _minutes_to_time(self, minutes: float) -> str:
        """Конвертация минут в время."""
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours:02d}:{mins:02d}"

    def _analyze_breaks(self, breaks: List[List[int]]) -> List[int]:
        """Анализ оптимальных перерывов."""
        if not breaks:
            return [15, 30, 45]  # Значения по умолчанию
        flat_breaks = [b for break_list in breaks for b in break_list]
        return sorted(list(set([int(np.mean(flat_breaks))])))

    def _calculate_confidence(self, patterns: List[Dict[str, Any]]) -> float:
        """Расчет уверенности в рекомендациях."""
        return min(
            1.0, len(patterns) / 10.0
        )  # Максимальная уверенность при 10+ паттернах

    def _prepare_interaction_text(self, data: Dict[str, Any]) -> str:
        """Подготовка текста для создания эмбеддинга."""
        parts = [
            data.get("learning_style", ""),
            " ".join(data.get("topics", [])),
            " ".join(data.get("successful_patterns", [])),
            " ".join(data.get("preferred_content_types", [])),
        ]
        return " ".join(filter(None, parts))
