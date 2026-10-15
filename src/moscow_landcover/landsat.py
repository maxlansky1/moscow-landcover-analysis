"""Выбор коллекции Landsat по году (бывшая E7, фрагмент)."""
from __future__ import annotations

from typing import Optional

from .config import AnalysisConfig


def build_collection(satellite_ids):
    """Объединяет несколько коллекций Landsat в одну через merge()."""
    import ee

    col = ee.ImageCollection(satellite_ids[0])
    for cid in satellite_ids[1:]:
        col = col.merge(ee.ImageCollection(cid))
    return col


def satellites_for_year(year: int, analysis: Optional[AnalysisConfig] = None):
    """Возвращает список id коллекций для года (L5/7 до 2012, L8/9 с 2013)."""
    analysis = analysis or AnalysisConfig()
    return list(analysis.satellites_early if year <= 2012 else analysis.satellites_late)


def build_collection_for_year(year: int, analysis: Optional[AnalysisConfig] = None):
    """Возвращает (merged_collection, sat_ids) для указанного года."""
    sat_ids = satellites_for_year(year, analysis)
    return build_collection(sat_ids), sat_ids
