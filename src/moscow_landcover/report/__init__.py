"""Итоговая сводка метрик (H2): публичный API `summary`.

Пакет разбит на подмодули: `logging` (таблицы и сборка), `plots` (графики),
`verdict` (успехи/проблемы), `cumulative` (накопительная таблица).
"""
from __future__ import annotations

from ..config import Config
from ._base import _normalize_results
from .logging import summary

__all__ = ["summary", "_normalize_results", "Config"]
