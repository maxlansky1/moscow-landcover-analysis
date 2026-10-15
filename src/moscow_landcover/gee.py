"""Аутентификация, инициализация и проверка Google Earth Engine (бывшие B1–B3)."""
from __future__ import annotations

import os
from typing import Optional

from loguru import logger

# Центр Москвы (Красная площадь) — для диагностической проверки.
MOSCOW_CENTER = [55.7558, 37.6173]          # [lat, lon]
MOSCOW_POINT_COORDS = [37.6173, 55.7558]    # (lon, lat)
DEFAULT_CHECK_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
DEFAULT_CHECK_DATE = ("2023-06-01", "2023-08-31")
CLOUD_THRESHOLDS = [5, 10, 20, 40, None]


def authenticate(auth_mode: str = "colab") -> None:
    """Аутентификация в Earth Engine (в Colab — `auth_mode='colab'`)."""
    import ee

    logger.info("=" * 70)
    logger.info("АУТЕНТИФИКАЦИЯ В GOOGLE EARTH ENGINE")
    logger.info("=" * 70)
    try:
        ee.Authenticate(auth_mode=auth_mode)
        logger.success("✅ Аутентификация прошла успешно. Токен сохранён.")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"❌ Ошибка при аутентификации: {exc}")
        raise


def resolve_project(project_id: Optional[str] = None) -> str:
    """Определяет id проекта: аргумент → MLC_EE_PROJECT → EE_PROJECT."""
    return (
        project_id
        or os.environ.get("MLC_EE_PROJECT")
        or os.environ.get("EE_PROJECT")
        or ""
    )


def initialize(project_id: Optional[str] = None) -> str:
    """Инициализирует клиент GEE с указанием проекта. Возвращает project_id."""
    import ee

    project_id = resolve_project(project_id)
    logger.info("=" * 70)
    logger.info("ИНИЦИАЛИЗАЦИЯ EARTH ENGINE")
    logger.info(f"Проект: {project_id or '(по умолчанию)'}")
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        logger.success(f"✅ Earth Engine инициализирован (проект: {project_id or 'default'})")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"❌ Ошибка при инициализации: {exc}")
        raise
    return project_id


def check(collection_id: str = DEFAULT_CHECK_COLLECTION,
          point_coords: Optional[list] = None,
          date_range: tuple[str, str] = DEFAULT_CHECK_DATE) -> dict:
    """Проверяет доступ: серверное вычисление, коллекция, тестовый снимок.

    Возвращает dict с результатами; не строит карту (это делает viz).
    """
    import ee

    point_coords = point_coords or MOSCOW_POINT_COORDS
    point = ee.Geometry.Point(list(point_coords))

    result: dict = {}
    logger.info("=" * 70)
    logger.info("ПРОВЕРКА РАБОТЫ EARTH ENGINE")
    logger.info("=" * 70)

    # 1. Серверное вычисление
    value = ee.Number(2).add(3).getInfo()
    result["server_ok"] = (value == 5)
    logger.info(f"  Серверное вычисление: 2 + 3 = {value}")

    # 2. Доступ к коллекции
    count = ee.ImageCollection(collection_id).size().getInfo()
    result["collection"] = collection_id
    result["collection_size"] = count
    logger.info(f"  Снимков в {collection_id}: {count:,}")

    # 3. Поиск тестового снимка с ослаблением порога облачности
    base = ee.ImageCollection(collection_id).filterBounds(point).filterDate(*date_range)
    scene = None
    used_threshold = None
    for threshold in CLOUD_THRESHOLDS:
        candidate = base.filter(ee.Filter.lt("CLOUD_COVER", threshold)) \
            if threshold is not None else base
        if candidate.size().getInfo() > 0:
            scene = candidate.first()
            used_threshold = threshold
            break

    if scene is None:
        raise RuntimeError("Не найдено ни одного тестового снимка над Москвой.")

    info = scene.getInfo()
    result["scene_id"] = info["id"]
    result["scene_date"] = info["properties"].get("DATE_ACQUIRED")
    result["scene_cloud"] = info["properties"].get("CLOUD_COVER")
    result["cloud_threshold"] = used_threshold
    logger.success("✅ Earth Engine работает корректно.")
    return result
