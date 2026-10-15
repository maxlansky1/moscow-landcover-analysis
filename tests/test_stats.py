"""Тесты форматтеров, метрик и истории прогонов (чистая логика)."""
from __future__ import annotations

import pandas as pd

from moscow_landcover import stats
from moscow_landcover.config import AnalysisConfig


def test_accuracy_marks_thresholds():
    assert stats.get_accuracy_mark(0.90, "oa") == "🟢"
    assert stats.get_accuracy_mark(0.80, "oa") == "🟡"
    assert stats.get_accuracy_mark(0.50, "oa") == "🔴"
    assert stats.get_accuracy_mark(0.85, "kappa") == "🟢"


def test_compute_per_class_metrics_perfect():
    matrix = [[10, 0, 0, 0], [0, 5, 0, 0], [0, 0, 8, 0], [0, 0, 0, 2]]
    metrics = stats.compute_per_class_metrics(matrix, ["a", "b", "c", "d"])
    for m in metrics.values():
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0


def test_compute_per_class_metrics_errors():
    matrix = [[8, 2, 0, 0], [0, 5, 0, 0], [0, 0, 8, 0], [0, 0, 0, 2]]
    metrics = stats.compute_per_class_metrics(matrix, ["a", "b", "c", "d"])
    assert metrics["a"]["precision"] == 1.0
    assert metrics["a"]["recall"] == 0.8
    assert 0 < metrics["a"]["f1"] < 1


def test_compute_accuracy_metrics_perfect():
    matrix = [[10, 0, 0, 0], [0, 5, 0, 0], [0, 0, 8, 0], [0, 0, 0, 2]]
    oa, kappa = stats.compute_accuracy_metrics(matrix)
    assert oa == 1.0
    assert kappa == 1.0


def test_compute_accuracy_metrics_matches_matrix():
    # согласованность OA/Kappa с матрицей (регресс на рассинхрон в pipeline)
    matrix = [[440, 68, 67, 9], [40, 494, 43, 4], [57, 33, 473, 9], [15, 9, 44, 540]]
    oa, kappa = stats.compute_accuracy_metrics(matrix)
    assert abs(oa - 1947 / 2345) < 1e-12
    assert 0.73 < kappa < 0.79


def test_format_areas_table_totals():
    analysis = AnalysisConfig()
    areas = {0: 100.0, 1: 50.0, 2: 25.0, 3: 25.0}
    table = stats.format_areas_table(areas, 200.0, analysis)
    assert "ИТОГО" in table
    assert "км²" in table
    for name in analysis.classes_ru:
        assert name in table


def test_format_confusion_matrix_shape():
    matrix = [[1, 2], [3, 4]]
    text = stats.format_confusion_matrix(matrix, ["0", "1"])
    assert "1" in text and "4" in text


def test_history_roundtrip(tmp_path):
    history_file = tmp_path / "run_history.json"
    cumulative = tmp_path / "cumulative.csv"
    result = {"oa": 0.9, "kappa": 0.85, "total_area": 1000.0,
              "n_scenes": 5, "elapsed_sec": 12.0,
              "areas": {0: 100.0, 1: 200.0, 2: 300.0, 3: 400.0}}
    record = stats.append_to_history("Historical", 2000, result,
                                     history_file, cumulative)
    assert record["territory"] == "Historical"

    loaded = stats.load_run_history(history_file)
    assert len(loaded) == 1
    assert loaded[0]["year"] == 2000

    df = pd.read_csv(cumulative)
    assert df.iloc[0]["Water_km2"] == 400.0

    # повторная запись того же (territory, year) не дублирует строку
    stats.append_to_history("Historical", 2000, result, history_file, cumulative)
    df2 = pd.read_csv(cumulative)
    assert len(df2) == 1


def test_format_full_stats_contains_sections():
    result = {
        "oa": 0.88, "kappa": 0.82, "areas": {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0},
        "total_area": 10.0, "n_scenes": 7, "train_size": 100,
        "test_size": 40, "elapsed_sec": 5.0,
        "confusion_matrix": [[1, 0, 0, 0], [0, 1, 0, 0],
                             [0, 0, 1, 0], [0, 0, 0, 1]],
    }
    text = stats.format_full_stats(result, "Historical", 2000, AnalysisConfig())
    assert "ПРОГОН" in text
    assert "OA" in text
    assert "Матрица ошибок" in text
