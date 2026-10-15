"""moscow_landcover — анализ динамики земного покрова Москвы по Landsat/GEE.

Публичный API собирает ключевые функции пайплайна и конфигурацию.
Тяжёлые GEE-модули (`ee`, `geemap`) импортируются лениво: сам пакет
импортируется без них (важно для тестов и окружений без Earth Engine).
"""
from __future__ import annotations

from .config import (
    TERRITORIES,
    AnalysisConfig,
    Config,
    DiagConfig,
    MapConfig,
    StatsConfig,
    build_paths,
    load_config,
    paths_from_config,
)
from .logging import get_logger, run_timestamp, setup_logging, start_run_log

__version__ = "1.1.0"

__all__ = [
    "TERRITORIES",
    "AnalysisConfig",
    "MapConfig",
    "StatsConfig",
    "DiagConfig",
    "Config",
    "build_paths",
    "load_config",
    "paths_from_config",
    "setup_logging",
    "start_run_log",
    "run_timestamp",
    "get_logger",
    "__version__",
]
