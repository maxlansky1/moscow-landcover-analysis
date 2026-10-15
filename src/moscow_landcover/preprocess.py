"""Предобработка снимков Landsat: маска, гармонизация каналов, калибровка.

Бывшая ячейка E1. Формула калибровки Collection 2 Level 2:
reflectance = DN * 0.0000275 - 0.2 (единый коэффициент для L5/L7/L8/L9).
"""
from __future__ import annotations

from typing import Optional

from .config import AnalysisConfig

# Значения по умолчанию берутся из единого источника — AnalysisConfig.
_SR = AnalysisConfig()
SR_SCALE_FACTOR = _SR.sr_scale_factor
SR_OFFSET = _SR.sr_offset

# Соответствие «сырых» каналов Landsat общим именам.
COMMON_BANDS = list(_SR.spectral_bands)
L57_RAW_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
L89_RAW_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
L57_SPACECRAFT = ["LANDSAT_5", "LANDSAT_7"]


def band_mapping(spacecraft: str) -> dict:
    """Чистая функция: общее имя → «сырое» имя канала для данного аппарата.

    Для L5/L7 и L8/L9 номера каналов различаются; L5/L7 — частный случай.
    """
    raw = L57_RAW_BANDS if spacecraft in L57_SPACECRAFT else L89_RAW_BANDS
    return dict(zip(COMMON_BANDS, raw))


def needs_l57_renaming(spacecraft: str) -> bool:
    """True, если снимок снят L5/L7 (каналы SR_B1..SR_B5, SR_B7)."""
    return spacecraft in L57_SPACECRAFT


def mask_landsat_sr(image):
    """Маскирует облака, тени и полосы SLC-off через канал QA_PIXEL.

    Биты: 0 — заполненные (Landsat 7 SLC-off), 1 — dilated cloud,
    3 — облака, 4 — тень от облаков. Для L5/L8/L9 бит 0 всегда 0.
    """
    qa = image.select("QA_PIXEL")
    mask = (
        qa.bitwiseAnd(1 << 0).eq(0)
        .And(qa.bitwiseAnd(1 << 1).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    )
    return image.updateMask(mask)


def harmonize_landsat_bands(image):
    """Переименовывает каналы L5/L7 и L8/L9 в общие имена. Значения не меняет."""
    import ee

    spacecraft = ee.String(image.get("SPACECRAFT_ID"))
    is_l57 = ee.List(L57_SPACECRAFT).contains(spacecraft)

    l57_bands = image.select(L57_RAW_BANDS, COMMON_BANDS)
    l89_bands = image.select(L89_RAW_BANDS, COMMON_BANDS)
    return ee.Image(ee.Algorithms.If(is_l57, l57_bands, l89_bands))


def calibrate_landsat_sr(image,
                         scale_factor: float = SR_SCALE_FACTOR,
                         offset: float = SR_OFFSET):
    """Переводит DN в отражательную способность (Collection 2 Level 2)."""
    return image.multiply(scale_factor).add(offset)


def preprocess_landsat(image,
                       scale_factor: float = SR_SCALE_FACTOR,
                       offset: float = SR_OFFSET):
    """Полная предобработка одного снимка: mask → harmonize → calibrate.

    Возвращает изображение с 6 каналами: blue, green, red, nir, swir1, swir2.
    """
    return calibrate_landsat_sr(
        harmonize_landsat_bands(mask_landsat_sr(image)),
        scale_factor=scale_factor,
        offset=offset,
    )
