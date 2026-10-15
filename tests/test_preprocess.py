"""Тесты чистых функций предобработки (без GEE)."""
from __future__ import annotations

from moscow_landcover import preprocess


def test_band_mapping_l89():
    mapping = preprocess.band_mapping("LANDSAT_8")
    assert mapping["blue"] == "SR_B2"
    assert mapping["nir"] == "SR_B5"
    assert mapping["swir2"] == "SR_B7"


def test_band_mapping_l57():
    mapping = preprocess.band_mapping("LANDSAT_5")
    assert mapping["blue"] == "SR_B1"
    assert mapping["nir"] == "SR_B4"
    assert mapping["swir1"] == "SR_B5"
    assert mapping["swir2"] == "SR_B7"


def test_needs_l57_renaming():
    assert preprocess.needs_l57_renaming("LANDSAT_5")
    assert preprocess.needs_l57_renaming("LANDSAT_7")
    assert not preprocess.needs_l57_renaming("LANDSAT_8")
    assert not preprocess.needs_l57_renaming("LANDSAT_9")


def test_common_band_order_and_length():
    assert preprocess.COMMON_BANDS == ["blue", "green", "red", "nir",
                                       "swir1", "swir2"]
    assert len(preprocess.L57_RAW_BANDS) == 6
    assert len(preprocess.L89_RAW_BANDS) == 6
