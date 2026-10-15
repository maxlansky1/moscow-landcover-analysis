"""Обучение Random Forest и парсинг важности признаков (бывшая E7/H1)."""
from __future__ import annotations

from typing import Optional

from loguru import logger

from .config import AnalysisConfig


def train_classifier(train_samples, analysis: Optional[AnalysisConfig] = None):
    """Обучает smileRandomForest на тренировочной выборке."""
    import ee

    analysis = analysis or AnalysisConfig()
    return ee.Classifier.smileRandomForest(
        numberOfTrees=analysis.rf_trees,
        minLeafPopulation=analysis.rf_min_leaf,
        bagFraction=analysis.rf_bag_fraction,
    ).train(
        features=train_samples,
        classProperty="class",
        inputProperties=analysis.features,
    )


def parse_feature_importance(classifier, features, territory_name: str = "",
                             year: Optional[int] = None) -> Optional[dict]:
    """Извлекает важность признаков из classifier.explain().

    Поддерживает несколько форматов GEE:
      1. {'importance': {feature: value}}          — актуальный
      2. {'names': [...], 'importance': [...]}
      3. {'importance': [...]} длиной len(features)
    Возвращает dict {feature: importance} или None.
    """
    try:
        explain_info = classifier.explain().getInfo()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[{territory_name} {year}] explain().getInfo() упал: {exc}")
        return None

    if not isinstance(explain_info, dict):
        return None

    importance = explain_info.get("importance")
    names = explain_info.get("names")

    # Вариант A: готовый dict
    if isinstance(importance, dict) and importance:
        try:
            return {k: float(v) for k, v in importance.items()}
        except (ValueError, TypeError):
            logger.warning(f"[{territory_name} {year}] importance dict содержит "
                           f"нечисловые значения")
            return None

    # Вариант B: два списка одинаковой длины
    if (isinstance(names, list) and isinstance(importance, list)
            and len(names) == len(importance) > 0):
        try:
            return {str(n): float(v) for n, v in zip(names, importance)}
        except (ValueError, TypeError):
            pass

    # Вариант C: только важности, порядок = features
    if (not names and isinstance(importance, list)
            and len(importance) == len(features)):
        try:
            return {f: float(v) for f, v in zip(features, importance)}
        except (ValueError, TypeError):
            pass

    logger.warning(f"[{territory_name} {year}] feature importance: "
                   f"неизвестная структура (names={names!r}, "
                   f"importance type={type(importance).__name__})")
    return None


def top_features(importance: dict, n: int = 5):
    """Топ-n признаков по важности (список пар)."""
    return sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:n]
