"""Обучающие точки на основе ESA WorldCover 2021 (бывшая E5/E6).

Метки берутся из ESA WorldCover (независимая готовая классификация),
признаки — из Landsat. Спектры ESA и Landsat не смешиваются.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from .config import AnalysisConfig


def get_class_areas(image, region, scale: int = 10) -> dict:
    """Возвращает {class_id: площадь_км²} через pixelArea + group."""
    import ee

    area_image = ee.Image.pixelArea().divide(1e6).addBands(image)
    areas = area_image.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()
    return {int(item["class"]): item["sum"] for item in areas.get("groups", [])}


def generate_training_data(boundary, year: int,
                           analysis: Optional[AnalysisConfig] = None,
                           boundary_name: str = "",
                           points_per_class: Optional[int] = None,
                           verbose: bool = True):
    """Генерирует стратифицированные обучающие точки из ESA WorldCover.

    Возвращает (points, info): points — ee.FeatureCollection с колонкой
    'class', info — словарь метаданных. При ошибке points=None.
    """
    import ee

    analysis = analysis or AnalysisConfig()
    points_per_class = points_per_class or analysis.points_per_class
    tag = f"[TrainingData:{boundary_name}]" if boundary_name else "[TrainingData]"

    info = {
        "boundary_name": boundary_name,
        "year": year,
        "training_year": analysis.training_year,
        "points_per_class": points_per_class,
        "total_points": 0,
        "class_distribution": {},
        "esa_areas_mapped": {},
        "esa_total_area": 0.0,
        "boundary_area": 0.0,
        "area_discrepancy_pct": 0.0,
    }

    if verbose:
        logger.info(f"{tag} Год {year}: эталон — ESA WorldCover {analysis.training_year}.")

    try:
        worldcover = (
            ee.ImageCollection(analysis.training_source)
            .filterDate(f"{analysis.training_year}-01-01", f"{analysis.training_year}-12-31")
            .first()
            .clip(boundary)
        )
        remapped = worldcover.remap(
            analysis.esa_remap_from, analysis.esa_remap_to, analysis.esa_remap_nodata
        ).rename("class")
        remapped = remapped.updateMask(remapped.neq(analysis.esa_remap_nodata))
    except Exception as exc:  # noqa: BLE001
        logger.error(f"{tag} Ошибка подготовки ESA: {exc}")
        return None, info

    # Площади ESA в границах (не фатально)
    try:
        areas_mapped = get_class_areas(remapped, boundary, scale=analysis.training_scale)
        info["esa_areas_mapped"] = areas_mapped
        info["esa_total_area"] = sum(areas_mapped.values())
        boundary_area_km2 = boundary.geometry().area(maxError=1).getInfo() / 1e6
        info["boundary_area"] = boundary_area_km2
        if boundary_area_km2 > 0:
            info["area_discrepancy_pct"] = abs(
                info["esa_total_area"] - boundary_area_km2
            ) / boundary_area_km2 * 100
        if verbose:
            logger.info(f"{tag} ESA в границах: {info['esa_total_area']:.1f} км² "
                        f"(граница {boundary_area_km2:.1f} км², "
                        f"Δ {info['area_discrepancy_pct']:.2f}%)")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"{tag} Ошибка проверки площадей: {exc}")

    try:
        training_points = remapped.stratifiedSample(
            numPoints=points_per_class,
            classBand="class",
            region=boundary,
            scale=analysis.training_scale,
            geometries=True,
            dropNulls=True,
            seed=analysis.random_seed,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"{tag} Ошибка stratifiedSample: {exc}")
        return None, info

    try:
        count = training_points.size().getInfo()
        dist_raw = training_points.aggregate_histogram("class").getInfo()
        dist: dict = {}
        for k, v in dist_raw.items():
            try:
                dist[int(float(k))] = int(v)
            except (ValueError, TypeError):
                logger.warning(f"{tag} Некорректный ключ класса: {k!r}")
        info["total_points"] = count
        info["class_distribution"] = dist

        if verbose:
            logger.info(f"{tag} Сгенерировано {count} точек "
                        f"(запрошено {points_per_class * analysis.n_classes}):")
            for cls_id in sorted(dist.keys()):
                name = analysis.classes[cls_id] if 0 <= cls_id < analysis.n_classes else f"?{cls_id}"
                logger.info(f"{tag}   класс {cls_id} ({name}): {dist[cls_id]} точек")
        if count == 0:
            logger.error(f"{tag} stratifiedSample вернул 0 точек.")
            return None, info
    except Exception as exc:  # noqa: BLE001
        logger.error(f"{tag} Ошибка при подсчёте точек: {exc}")
        return None, info

    if verbose:
        logger.success(f"{tag} Год {year}: обучающие точки готовы.")
    return training_points, info
