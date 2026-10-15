"""Диагностика качества данных (бывшие F1/F2).

F1 — доступность снимков Landsat за 2000–2026 по обеим территориям.
F2 — качество летних композитов и обучающих точек.
"""
from __future__ import annotations

from .availability import (
    availability,
    build_scene_summary,
    get_all_scenes_metadata,
)
from .spectral import (
    compute_composite_percentiles,
    compute_composite_stats_light,
    compute_training_validation,
    run_diagnostics,
    subsample_points_tiered,
)

__all__ = [
    "availability",
    "build_scene_summary",
    "get_all_scenes_metadata",
    "compute_composite_percentiles",
    "compute_composite_stats_light",
    "compute_training_validation",
    "run_diagnostics",
    "subsample_points_tiered",
]
