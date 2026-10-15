"""Медианный летний композит Landsat с 15 признаками (бывшая E3/E4).

Признаки: 6 спектральных каналов + 7 индексов + 2 текстурных (GLCM).
GLCM (glcmTexture) работает только с 8-битными изображениями и возвращает
каналы с сокращёнными суффиксами, поэтому энтропия — это `nir_ent`.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from .config import AnalysisConfig
from .preprocess import preprocess_landsat

# Индексы, вычисляемые из каналов композита (порядок важен для FEATURES).
INDEX_NAMES = ["NDVI", "NDWI", "NDBI", "BSI", "SAVI", "NDMI", "MNDWI"]


def _add_indices(composite):
    """Добавляет 7 индексов к композиту (с clamp(-1, 1))."""
    blue = composite.select("blue")
    green = composite.select("green")
    red = composite.select("red")
    nir = composite.select("nir")
    swir1 = composite.select("swir1")

    ndvi = nir.subtract(red).divide(nir.add(red)).clamp(-1, 1).rename("NDVI")
    ndwi = green.subtract(nir).divide(green.add(nir)).clamp(-1, 1).rename("NDWI")
    ndbi = swir1.subtract(nir).divide(swir1.add(nir)).clamp(-1, 1).rename("NDBI")
    bsi = (
        swir1.add(red).subtract(nir.add(blue))
        .divide(swir1.add(red).add(nir).add(blue))
        .clamp(-1, 1)
        .rename("BSI")
    )
    savi = (
        nir.subtract(red).divide(nir.add(red).add(0.5))
        .multiply(1.5).clamp(-1, 1).rename("SAVI")
    )
    ndmi = nir.subtract(swir1).divide(nir.add(swir1)).clamp(-1, 1).rename("NDMI")
    mndwi = green.subtract(swir1).divide(green.add(swir1)).clamp(-1, 1).rename("MNDWI")
    return [ndvi, ndwi, ndbi, bsi, savi, ndmi, mndwi]


def add_texture_bands(composite, size: int = 3, verbose: bool = False,
                      tag: str = ""):
    """Вычисляет GLCM-текстуру на канале NIR → (NIR_contrast, NIR_entropy).

    glcmTexture возвращает `nir_contrast` и `nir_ent` (сокращённый суффикс).
    """
    nir_8bit = composite.select("nir").multiply(255).clamp(0, 255).toByte()
    glcm = nir_8bit.glcmTexture(size=size)
    if verbose:
        logger.debug(f"{tag} GLCM bands = {glcm.bandNames().getInfo()}")
    contrast = glcm.select("nir_contrast").rename("NIR_contrast")
    entropy = glcm.select("nir_ent").rename("NIR_entropy")
    return contrast, entropy


def create_composite(collection, boundary, year: int, territory_name: str,
                     analysis: Optional[AnalysisConfig] = None,
                     min_images: Optional[int] = None,
                     verbose: bool = True):
    """Создаёт медианный композит с 15 признаками за лето `year`.

    Возвращает (composite, info) или (None, info). `info` содержит
    target_year/used_year/n_scenes/fallback_used.
    """
    analysis = analysis or AnalysisConfig()
    min_images = min_images if min_images is not None else analysis.min_images_for_composite
    tag = f"[{territory_name}]"

    def _preprocess(img):
        return preprocess_landsat(img, analysis.sr_scale_factor, analysis.sr_offset)

    def _filter(y):
        return (
            collection
            .filterBounds(boundary)
            .filterDate(f"{y}-{analysis.season_start}", f"{y}-{analysis.season_end}")
            .map(_preprocess)
        )

    filtered = _filter(year)
    count = filtered.size().getInfo()
    info = {"target_year": year, "used_year": year, "n_scenes": count,
            "fallback_used": False}

    if verbose:
        logger.info(f"{tag} Год {year}: найдено {count} снимков.")

    if count < min_images:
        if verbose:
            logger.warning(f"{tag} Год {year}: снимков мало (<{min_images}), "
                           f"ищу соседние годы {analysis.fallback_offsets}...")
        for offset in analysis.fallback_offsets:
            adj_year = year + offset
            adj = _filter(adj_year)
            adj_count = adj.size().getInfo()
            if verbose:
                logger.info(f"{tag} Год {year}: проверяю {adj_year} — {adj_count} снимков.")
            if adj_count >= min_images:
                filtered, count = adj, adj_count
                info.update(used_year=adj_year, n_scenes=adj_count, fallback_used=True)
                break
        if count == 0:
            logger.error(f"{tag} Год {year}: нет снимков совсем.")
            return None, info

    composite = filtered.median().clip(boundary)

    band_names = composite.bandNames().getInfo()
    missing = [b for b in analysis.spectral_bands if b not in band_names]
    if missing:
        logger.error(f"{tag} Год {year}: в композите нет каналов {missing}. "
                     f"Есть: {band_names}")
        return None, info

    composite = composite.addBands(_add_indices(composite))
    contrast, entropy = add_texture_bands(composite, verbose=verbose, tag=tag)
    composite = composite.addBands([contrast, entropy])

    final_bands = composite.bandNames().getInfo()
    if len(final_bands) != len(analysis.features):
        logger.warning(f"{tag} Год {year}: в композите {len(final_bands)} каналов, "
                       f"ожидалось {len(analysis.features)}. Список: {final_bands}")
    elif verbose:
        logger.success(f"{tag} Год {year}: композит создан из {count} снимков "
                       f"({len(final_bands)} признаков).")
    return composite, info
