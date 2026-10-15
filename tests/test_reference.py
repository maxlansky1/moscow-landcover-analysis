"""Тесты эталонного прогона: заморозка и сравнение (чистая логика)."""
from __future__ import annotations

from moscow_landcover import reference


def _result(oa, kappa, areas):
    return {"oa": oa, "kappa": kappa, "areas": areas, "total_area": sum(areas.values())}


def test_freeze_and_load_roundtrip(tmp_path):
    path = tmp_path / "reference.json"
    results = {"Historical": {2000: _result(0.8, 0.7, {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0})}}
    reference.freeze_reference(results, path)
    loaded = reference.load_reference(path)
    assert loaded["Historical"][2000]["oa"] == 0.8


def test_load_reference_missing_returns_none(tmp_path):
    assert reference.load_reference(tmp_path / "nope.json") is None


def test_compare_ok_within_tolerance():
    ref = {"Historical": {2000: _result(0.80, 0.70, {0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0})}}
    cur = {"Historical": {2000: _result(0.805, 0.702, {0: 10.1, 1: 20.0, 2: 30.0, 3: 40.0})}}
    out = reference.compare(cur, ref, verbose=False)
    assert out["status"] == "ok"
    assert out["checked"] == 1


def test_compare_flags_mismatch():
    ref = {"Historical": {2000: _result(0.80, 0.70, {0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0})}}
    cur = {"Historical": {2000: _result(0.60, 0.70, {0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0})}}
    out = reference.compare(cur, ref, verbose=False)
    assert out["status"] == "mismatch"
    assert out["mismatches"][0]["field"] == "oa"


def test_default_reference_path():
    paths = {"INTERMEDIATE": "/tmp/inter"}
    assert str(reference.default_reference_path(paths)).endswith("reference.json")
