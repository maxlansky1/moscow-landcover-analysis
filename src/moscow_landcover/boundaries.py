"""Загрузка административных границ Москвы из GeoJSON (бывшая D1)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger

# Ожидаемые площади (км²) из спецификации — для контроля вакцидности.
EXPECTED_AREAS = {
    "Historical": 1062,
    "Greater": 2514,
}

BOUNDARY_FILENAMES = {
    "Historical": "moscow_historical.geojson",
    "Greater": "moscow_greater.geojson",
}


def load_boundary(path: Path | str, name: str = "",
                  expected_area_km2: Optional[float] = None):
    """Загружает одну границу: GeoJSON → ee.FeatureCollection → union().

    Возвращает (boundary_fc, area_km2).
    """
    import ee

    path = Path(path)
    tag = f"[{name}]" if name else ""
    if not path.exists():
        logger.error(f"{tag} ❌ Файл не найден: {path}")
        raise FileNotFoundError(f"Не найден {path.name}")

    with open(path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    fc = ee.FeatureCollection(geojson)
    n_features = fc.size().getInfo()
    boundary = fc.union()
    logger.info(f"{tag} GeoJSON прочитан: {n_features} объектов, union() применён.")

    area_km2 = boundary.geometry().area(maxError=1).getInfo() / 1e6
    if expected_area_km2:
        diff_pct = abs(area_km2 - expected_area_km2) / expected_area_km2 * 100
        level = logger.success if diff_pct < 2.0 else logger.warning
        level(f"{tag} Площадь {area_km2:,.1f} км² "
              f"(ожидалось {expected_area_km2:,}, Δ {diff_pct:.2f}%)")
    else:
        logger.info(f"{tag} Площадь: {area_km2:,.1f} км²")
    return boundary, area_km2


def load_boundaries(paths: dict, names=("Historical", "Greater"),
                    expected_areas: Optional[dict] = None):
    """Загружает границы по словарю путей PATHS.

    Возвращает (boundaries, areas) — словари {name: ee.FeatureCollection}
    и {name: area_km2}.
    """
    expected_areas = expected_areas or EXPECTED_AREAS
    boundaries: dict = {}
    areas: dict = {}

    logger.info("=" * 70)
    logger.info("ЗАГРУЗКА ГРАНИЦ МОСКВЫ")
    logger.info("=" * 70)

    for name in names:
        path = paths["BOUNDARIES"] / BOUNDARY_FILENAMES[name]
        boundary, area = load_boundary(
            path, name=name, expected_area_km2=expected_areas.get(name)
        )
        boundaries[name] = boundary
        areas[name] = area

    _check_nesting(boundaries, areas)
    return boundaries, areas


def _check_nesting(boundaries: dict, areas: dict) -> None:
    """Проверяет, что Historical вложена в Greater."""
    if "Historical" not in boundaries or "Greater" not in boundaries:
        return
    try:
        intersection = boundaries["Historical"].geometry().intersection(
            boundaries["Greater"].geometry(), maxError=1
        )
        inter_area = intersection.area(maxError=1).getInfo() / 1e6
        hist_area = areas["Historical"]
        overlap_pct = inter_area / hist_area * 100 if hist_area > 0 else 0
        logger.info(f"Пересечение Historical∩Greater: {inter_area:,.1f} км² "
                    f"({overlap_pct:.2f}% исторической площади).")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Не удалось проверить вложенность границ: {exc}")
