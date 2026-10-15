"""Тесты чистой логики диагностики и отчёта (без GEE)."""
from __future__ import annotations

import pandas as pd

from moscow_landcover import diagnostics
from moscow_landcover.config import AnalysisConfig
from moscow_landcover.report import _normalize_results


def test_build_scene_summary_empty():
    df = diagnostics.build_scene_summary([])
    assert df.empty
    assert list(df.columns) == ["year", "n_scenes", "median_cloud",
                                "min_cloud", "max_cloud", "sensors"]


def test_build_scene_summary_aggregates():
    meta = [
        {"year": 2000, "month": 6, "cloud": 10.0, "sensor": "LANDSAT_5"},
        {"year": 2000, "month": 7, "cloud": 30.0, "sensor": "LANDSAT_7"},
        {"year": 2001, "month": 6, "cloud": 5.0, "sensor": "LANDSAT_5"},
    ]
    df = diagnostics.build_scene_summary(meta)
    assert set(df["year"]) == {2000, 2001}
    row = df[df["year"] == 2000].iloc[0]
    assert row["n_scenes"] == 2
    assert row["median_cloud"] == 20.0
    assert row["min_cloud"] == 10.0
    assert row["max_cloud"] == 30.0
    assert row["sensors"] == "LANDSAT_5,LANDSAT_7"


def test_subsample_uses_n_classes(monkeypatch):
    """subsample_points_tiered фильтрует по каждому классу (без ee)."""
    analysis = AnalysisConfig()

    calls = {"filters": 0}

    class _FakePoints:
        def filter(self, _):
            calls["filters"] += 1
            return self

        def randomColumn(self, *a, **k):
            return self

        def sort(self, *a, **k):
            return self

        def limit(self, *a, **k):
            return self

        def flatten(self):
            return self

    # подменяем ee.FeatureCollection, чтобы не строить реальный коллекцию
    import sys
    import types
    fake_ee = sys.modules.get("ee")
    if fake_ee is None:
        fake_ee = types.ModuleType("ee")
        sys.modules["ee"] = fake_ee
    fake_ee.FeatureCollection = lambda *a, **k: _FakePoints()
    fake_ee.Filter = type("Filter", (), {"eq": staticmethod(lambda *a, **k: None)})

    diagnostics.subsample_points_tiered(_FakePoints(), 10, n_classes=analysis.n_classes)
    assert calls["filters"] == analysis.n_classes


def test_normalize_results_converts_area_keys():
    raw = {
        "Historical": {
            "2000": {"oa": 0.7, "kappa": 0.6, "total_area": 100.0,
                     "confusion_matrix": [[1, 0, 0, 0], [0, 1, 0, 0],
                                          [0, 0, 1, 0], [0, 0, 0, 1]],
                     "areas": {"0": 1.0, "1": 2.0, "2": 3.0, "3": 4.0}},
        }
    }
    norm = _normalize_results(raw)
    result = norm["Historical"][2000]
    assert 0 in result["areas"]
    assert result["areas"][3] == 4.0


def test_report_verdict_runs_on_json_loaded_results(monkeypatch):
    """summary должна переваривать results_by_year без matplotlib/GEE."""
    from moscow_landcover import report
    from moscow_landcover.report import logging as report_logging

    monkeypatch.setattr(report_logging, "have_matplotlib", lambda: False)

    cfg = report.Config.standard(base="/tmp/does-not-matter")
    import types
    cfg.paths = {k: __import__("pathlib").Path("/tmp/h2")
                 for k in ("OUTPUT", "PLOTS", "TABLES", "INTERMEDIATE")}

    results = {
        "Historical": {
            2000: {"oa": 0.7130, "kappa": 0.6172, "total_area": 1062.0,
                   "areas": {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0},
                   "confusion_matrix": [[1, 0, 0, 0], [0, 1, 0, 0],
                                        [0, 0, 1, 0], [0, 0, 0, 1]],
                   "feature_importance": {}},
        },
    }
    out = report.summary(cfg, results, verbose=False, plots=False)
    assert "issues" in out and "successes" in out
