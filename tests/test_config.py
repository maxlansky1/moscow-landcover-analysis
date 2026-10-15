"""Тесты конфигурации и построения путей (чистая логика, без GEE)."""
from __future__ import annotations

from pathlib import Path

import pytest

from moscow_landcover.config import (
    AnalysisConfig,
    Config,
    build_paths,
    load_config,
)


def test_features_are_15_and_ordered():
    a = AnalysisConfig()
    assert len(a.features) == 15
    assert a.features == a.spectral_bands + a.index_bands + a.texture_bands
    assert a.n_classes == 4


def test_paths_contain_year_subdirs(tmp_path):
    a = AnalysisConfig()
    paths = build_paths(tmp_path, a)
    assert paths["BASE"] == tmp_path
    assert paths["OUT_HIST_2000"] == tmp_path / "02_Output" / "Historical_Moscow" / "2000"
    assert paths["OUT_GREATER_2026"] == tmp_path / "02_Output" / "Greater_Moscow" / "2026"
    assert "OUT_DIFF_Historical_2000_2026" in paths


def test_build_paths_create(tmp_path):
    a = AnalysisConfig()
    paths = build_paths(tmp_path, a, create=True)
    for p in paths.values():
        assert Path(p).is_dir()


def test_load_config_defaults():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.analysis.historical_years == [2000, 2007, 2013, 2019, 2026]
    assert cfg.map.territories["Historical"]["zoom"] == 10


def test_load_config_yaml_overrides(tmp_path):
    cfg_file = tmp_path / "local.yaml"
    cfg_file.write_text(
        "analysis:\n"
        "  rf_trees: 50\n"
        "  historical_years: [2000, 2026]\n"
        "paths:\n"
        f"  base: \"{tmp_path / 'proj'}\"\n"
        "gee:\n"
        "  project_id: my-project\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.analysis.rf_trees == 50
    assert cfg.analysis.historical_years == [2000, 2026]
    assert cfg.project_id == "my-project"
    assert cfg.paths["OUT_HIST_2000"] == tmp_path / "proj" / "02_Output" / "Historical_Moscow" / "2000"


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")
