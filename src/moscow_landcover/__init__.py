"""moscow_landcover — анализ динамики земного покрова Москвы по Landsat/GEE.

Публичный API собирает ключевые функции пайплайна и конфигурацию.
Тяжёлые GEE-модули (`ee`, `geemap`) импортируются лениво: сам пакет
импортируется без них (важно для тестов и окружений без Earth Engine).
"""
from __future__ import annotations

from .config import (
    AnalysisConfig,
    Config,
    MapConfig,
    StatsConfig,
    build_paths,
    load_config,
    paths_from_config,
)
from .logging import get_logger, run_timestamp, setup_logging

__version__ = "1.0.0"

__all__ = [
    "AnalysisConfig",
    "MapConfig",
    "StatsConfig",
    "Config",
    "build_paths",
    "load_config",
    "paths_from_config",
    "setup_logging",
    "run_timestamp",
    "get_logger",
    "__version__",
]
