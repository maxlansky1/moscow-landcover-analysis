"""Экспорт классификаций в GEE Assets / Google Drive (бывшая H3).

Классификации живут только в памяти сессии; эти функции запускают
batch-задачи, чтобы сохранить их надолго.
"""
from __future__ import annotations

from loguru import logger


def export_classifications_to_assets(results_by_year: dict, boundaries_map: dict,
                                     crs: str = "EPSG:32637",
                                     gee_folder: str = "moscow_landcover",
                                     gee_user: str = "your_username",
                                     scale: int = 30,
                                     max_pixels: float = 1e13,
                                     start: bool = True) -> list:
    """Запускает экспорт классификаций в GEE Assets.

    Возвращает список dict с territory/year/task_id/asset_id.
    """
    import ee

    folder = f"users/{gee_user}/{gee_folder}"
    tasks = []
    for territory_name, subset in results_by_year.items():
        boundary = boundaries_map[territory_name]
        for year, result in subset.items():
            asset_id = f"{folder}/{territory_name}_{year}"
            task = ee.batch.Export.image.toAsset(
                image=result["classified"].toInt(),
                description=f"moscow_lc_{territory_name}_{year}",
                assetId=asset_id,
                scale=scale,
                region=boundary.geometry(),
                maxPixels=max_pixels,
                crs=crs,
            )
            if start:
                task.start()
            tasks.append({"territory": territory_name, "year": year,
                          "task_id": task.id, "asset_id": asset_id})
            logger.info(f"Запущен экспорт: {asset_id} (task {task.id})")
    logger.success(f"Запущено {len(tasks)} задач экспорта (Assets).")
    return tasks


def export_classifications_to_drive(results_by_year: dict, boundaries_map: dict,
                                    drive_folder: str = "Moscow_LandCover_Classifications",
                                    crs: str = "EPSG:32637",
                                    scale: int = 30,
                                    max_pixels: float = 1e13,
                                    start: bool = True) -> list:
    """Запускает экспорт классификаций в Google Drive как GeoTIFF."""
    import ee

    tasks = []
    for territory_name, subset in results_by_year.items():
        boundary = boundaries_map[territory_name]
        for year, result in subset.items():
            task = ee.batch.Export.image.toDrive(
                image=result["classified"].toInt(),
                description=f"moscow_lc_{territory_name}_{year}",
                folder=drive_folder,
                fileNamePrefix=f"{territory_name}_{year}",
                scale=scale,
                region=boundary.geometry(),
                maxPixels=max_pixels,
                crs=crs,
            )
            if start:
                task.start()
            tasks.append({"territory": territory_name, "year": year,
                          "task_id": task.id})
            logger.info(f"Запущен экспорт: {territory_name}_{year} (task {task.id})")
    logger.success(f"Запущено {len(tasks)} задач экспорта (Drive).")
    return tasks
