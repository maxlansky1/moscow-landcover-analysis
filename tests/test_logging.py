"""Тесты раскладки логов по дням и меткам запуска (без GEE)."""
from __future__ import annotations

from datetime import datetime

from moscow_landcover.logging import setup_logging, start_run_log


def test_setup_logging_day_folder_and_label(tmp_path):
    log_file = setup_logging(tmp_path, label="smoke test")
    day = datetime.now().strftime("%Y-%m-%d")
    assert log_file.parent == tmp_path / day
    assert log_file.name.startswith("run_")
    assert log_file.name.endswith("_smoke_test.log")
    assert log_file.exists()


def test_start_run_log_uses_paths(tmp_path):
    paths = {"LOGS": tmp_path / "04_Logs"}
    log_file = start_run_log(paths, label="full")
    assert log_file.parent.parent == paths["LOGS"]
    assert log_file.name.endswith("_full.log")
