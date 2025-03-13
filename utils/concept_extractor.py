"""
Модуль для извлечения концепций, терминов и тегов из текста.
Использует SpaCy для NER и извлечения ключевых концепций.
"""

import logging
from typing import Dict, List, Set, Tuple
import spacy
from collections import Counter
from dataclasses import dataclass
import re


@dataclass
class ExtractedConcept:
    """Структура данных для хранения извлеченной концепции"""

    text: str
    type: str
    context: str
    importance: float
    frequency: int


class ConceptExtractor:
    def __init__(self):
        """Инициализация экстрактора концепций"""
        try:
            self.nlp = spacy.load("en_core_web_lg")
            # Добавляем специальные паттерны для образовательных терминов
            self.edu_patterns = [
                {
                    "label": "EDUCATIONAL_TERM",
                    "pattern": [
                        {"LOWER": {"REGEX": "(learn|study|concept|theory|principle)"}}
                    ],
                },
                {
                    "label": "EDUCATIONAL_TERM",
                    "pattern": [{"POS": "NOUN"}, {"LOWER": "theory"}],
                },
                {
                    "label": "EDUCATIONAL_TERM",
                    "pattern": [{"POS": "NOUN"}, {"LOWER": "concept"}],
                },
            ]
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(self.edu_patterns)

            logging.info("ConceptExtractor initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing ConceptExtractor: {str(e)}")
            raise

    def extract_concepts(self, text: str) -> List[ExtractedConcept]:
        """
        Извлечение концепций из текста

        Args:
            text: Исходный текст

        Returns:
            List[ExtractedConcept]: Список извлеченных концепций
        """
        try:
            doc = self.nlp(text)
            concepts = []

            # Извлекаем именованные сущности
            for ent in doc.ents:
                # Получаем контекст (предложение, содержащее сущность)
                context = next(
                    (
                        sent.text
                        for sent in doc.sents
                        if ent.start >= sent.start and ent.end <= sent.end
                    ),
                    "",
                )

                # Вычисляем важность на основе частоты и позиции
                importance = self._calculate_importance(ent, doc)

                concepts.append(
                    ExtractedConcept(
                        text=ent.text,
                        type=ent.label_,
                        context=context,
                        importance=importance,
                        frequency=1,
                    )
                )

            # Объединяем похожие концепции
            concepts = self._merge_similar_concepts(concepts)

            # Сортируем по важности
            concepts.sort(key=lambda x: x.importance, reverse=True)

            return concepts
        except Exception as e:
            logging.error(f"Error extracting concepts: {str(e)}")
            return []

    def extract_terms(self, text: str) -> Dict[str, str]:
        """
        Извлечение терминов и их определений

        Args:
            text: Исходный текст

        Returns:
            Dict[str, str]: Словарь терминов и их контекстов
        """
        try:
            doc = self.nlp(text)
            terms = {}

            # Ищем паттерны определений
            definition_patterns = [
                r"(?P<term>\w+)\s+is\s+(?P<definition>[^.]+)",
                r"(?P<term>\w+)\s+refers to\s+(?P<definition>[^.]+)",
                r"(?P<term>\w+)\s+means\s+(?P<definition>[^.]+)",
                r"(?P<term>\w+)\s+defined as\s+(?P<definition>[^.]+)",
            ]

            for pattern in definition_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    term = match.group("term")
                    definition = match.group("definition")
                    if term and definition:
                        terms[term] = definition.strip()

            # Добавляем термины из именованных сущностей
            for ent in doc.ents:
                if (
                    ent.label_ in ["EDUCATIONAL_TERM", "SCIENTIFIC_TERM"]
                    and ent.text not in terms
                ):
                    # Получаем контекст использования термина
                    context = next(
                        (
                            sent.text
                            for sent in doc.sents
                            if ent.start >= sent.start and ent.end <= sent.end
                        ),
                        "",
                    )
                    terms[ent.text] = context

            return terms
        except Exception as e:
            logging.error(f"Error extracting terms: {str(e)}")
            return {}

    def generate_tags(self, text: str, max_tags: int = 10) -> List[str]:
        """
        Генерация тегов из текста

        Args:
            text: Исходный текст
            max_tags: Максимальное количество тегов

        Returns:
            List[str]: Список тегов
        """
        try:
            doc = self.nlp(text)

            # Собираем все существительные и именованные сущности
            potential_tags = []

            # Добавляем именованные сущности
            potential_tags.extend([ent.text.lower() for ent in doc.ents])

            # Добавляем существительные
            potential_tags.extend(
                [token.text.lower() for token in doc if token.pos_ == "NOUN"]
            )

            # Считаем частоту
            tag_freq = Counter(potential_tags)

            # Фильтруем и сортируем
            filtered_tags = [
                tag
                for tag, freq in tag_freq.most_common(max_tags)
                if len(tag) > 2  # Исключаем слишком короткие теги
                and not tag.isdigit()  # Исключаем числа
            ]

            return filtered_tags[:max_tags]
        except Exception as e:
            logging.error(f"Error generating tags: {str(e)}")
            return []

    def _calculate_importance(
        self, entity: spacy.tokens.Span, doc: spacy.tokens.Doc
    ) -> float:
        """Вычисление важности концепции"""
        try:
            # Факторы важности:
            # 1. Позиция в тексте (концепции в начале важнее)
            position_weight = 1 - (entity.start / len(doc))

            # 2. Частота упоминания
            frequency = sum(1 for ent in doc.ents if ent.text == entity.text)
            frequency_weight = frequency / len(doc.ents) if doc.ents else 0

            # 3. Тип сущности
            type_weights = {
                "EDUCATIONAL_TERM": 1.0,
                "SCIENTIFIC_TERM": 0.9,
                "PERSON": 0.8,
                "ORG": 0.7,
                "GPE": 0.6,
                "DATE": 0.5,
                "CARDINAL": 0.4,
            }
            type_weight = type_weights.get(entity.label_, 0.3)

            # Комбинируем веса
            importance = (
                position_weight * 0.3 + frequency_weight * 0.4 + type_weight * 0.3
            )

            return importance
        except Exception as e:
            logging.error(f"Error calculating importance: {str(e)}")
            return 0.0

    def _merge_similar_concepts(
        self, concepts: List[ExtractedConcept]
    ) -> List[ExtractedConcept]:
        """Объединение похожих концепций"""
        try:
            if not concepts:
                return []

            merged = []
            used = set()

            for i, concept1 in enumerate(concepts):
                if i in used:
                    continue

                similar = [concept1]
                used.add(i)

                # Ищем похожие концепции
                for j, concept2 in enumerate(concepts[i + 1 :], i + 1):
                    if j not in used:
                        # Проверяем схожесть текста
                        similarity = self.nlp(concept1.text).similarity(
                            self.nlp(concept2.text)
                        )
                        if similarity > 0.8:  # Порог схожести
                            similar.append(concept2)
                            used.add(j)

                # Объединяем похожие концепции
                if len(similar) > 1:
                    merged.append(
                        ExtractedConcept(
                            text=max(similar, key=lambda x: len(x.text)).text,
                            type=similar[0].type,
                            context=max(similar, key=lambda x: len(x.context)).context,
                            importance=max(s.importance for s in similar),
                            frequency=sum(s.frequency for s in similar),
                        )
                    )
                else:
                    merged.append(similar[0])

            return merged
        except Exception as e:
            logging.error(f"Error merging similar concepts: {str(e)}")
            return concepts
