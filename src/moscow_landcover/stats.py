"""Форматирование и агрегация статистик, история прогонов (бывшие C3/H2).

Здесь — детерминированная логика: ASCII-таблицы, метрики из матрицы ошибок,
метки точности и накопительная таблица. Визуализация графиков — в `report`.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from .config import AnalysisConfig, StatsConfig


# ---------------------------------------------------------------------
# Метки точности
# ---------------------------------------------------------------------
def get_accuracy_mark(value: float, metric: str = "oa",
                      stats: Optional[StatsConfig] = None) -> str:
    """Возвращает метку 🟢/🟡/🔴 для значения метрики."""
    stats = stats or StatsConfig()
    for threshold, mark in stats.accuracy_marks[metric]:
        if value >= threshold:
            return mark
    return "⚪"


# ---------------------------------------------------------------------
# Форматтеры
# ---------------------------------------------------------------------
def format_confusion_matrix(matrix, classes=None,
                            stats: Optional[StatsConfig] = None) -> str:
    """ASCII-таблица матрицы ошибок."""
    stats = stats or StatsConfig()
    classes = classes or ["0", "1", "2", "3"]
    w = stats.format["table_col_width"]
    header = " " * 22 + "".join(f"{c.split()[0][:w - 2]:>{w}}" for c in classes)
    lines = [header, " " * 22 + "-" * (w * len(classes))]
    for i, row in enumerate(matrix):
        row_str = "".join(f"{int(v):>{w}}" for v in row)
        lines.append(f"  {classes[i][:20]:<20}{row_str}")
    return "\n".join(lines)


def format_areas_table(class_areas: dict, total_area: float,
                       analysis: Optional[AnalysisConfig] = None,
                       stats: Optional[StatsConfig] = None) -> str:
    """ASCII-таблица площадей по классам."""
    analysis = analysis or AnalysisConfig()
    stats = stats or StatsConfig()
    ap = stats.format["area_precision"]
    pp = stats.format["percent_precision"]
    lines = [f'  {"Класс":<32}{"Площадь":>12}{"Доля":>10}', "  " + "-" * 54]
    for i in range(analysis.n_classes):
        area = class_areas.get(i, 0)
        pct = (area / total_area * 100) if total_area > 0 else 0
        lines.append(f"  {analysis.classes_ru[i]:<32}{area:>{12}.{ap}f} км²"
                     f"{pct:>{9}.{pp}f}%")
    lines.append("  " + "-" * 54)
    lines.append(f'  {"ИТОГО":<32}{total_area:>{12}.{ap}f} км²{100.0:>{9}.{pp}f}%')
    return "\n".join(lines)


def format_accuracy(oa: float, kappa: float,
                    stats: Optional[StatsConfig] = None) -> str:
    """Строка с OA и Kappa + метки."""
    stats = stats or StatsConfig()
    fp = stats.format["float_precision"]
    return (f"  OA:     {oa:.{fp}f}  {get_accuracy_mark(oa, 'oa', stats)}\n"
            f"  Kappa:  {kappa:.{fp}f}  {get_accuracy_mark(kappa, 'kappa', stats)}")


def compute_per_class_metrics(matrix, classes=None) -> dict:
    """precision/recall/F1 из матрицы ошибок (matrix[i][j]: истина i, пред. j)."""
    classes = classes or ["0", "1", "2", "3"]
    n = len(matrix)
    metrics = {}
    for i in range(n):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(n) if j != i)
        fn = sum(matrix[i][j] for j in range(n) if j != i)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        metrics[classes[i]] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def compute_accuracy_metrics(matrix) -> tuple[float, float]:
    """OA и Cohen's kappa из матрицы ошибок (matrix[i][j]: истина i, пред. j).

    Считаются локально из ОДНОЙ матрицы, поэтому гарантированно согласованы
    с ней (в отличие от отдельных серверных `errorMatrix.accuracy()/kappa()`).
    """
    n_total = sum(sum(row) for row in matrix)
    if n_total == 0:
        return 0.0, 0.0
    n = len(matrix)
    oa = sum(matrix[i][i] for i in range(n)) / n_total
    row_sums = [sum(matrix[i]) for i in range(n)]
    col_sums = [sum(matrix[i][j] for i in range(n)) for j in range(n)]
    pe = sum(row_sums[i] * col_sums[i] for i in range(n)) / (n_total ** 2)
    kappa = (oa - pe) / (1 - pe) if pe < 1 else 0.0
    return oa, kappa


def format_per_class_metrics(metrics: dict,
                             stats: Optional[StatsConfig] = None) -> str:
    """ASCII-таблица per-class метрик."""
    stats = stats or StatsConfig()
    fp = stats.format["float_precision"]
    lines = [f'  {"Класс":<25}{"Precision":>12}{"Recall":>12}{"F1":>12}',
             "  " + "-" * 61]
    for name, m in metrics.items():
        lines.append(f'  {name:<25}{m["precision"]:>{12}.{fp}f}'
                     f'{m["recall"]:>{12}.{fp}f}{m["f1"]:>{12}.{fp}f}')
    return "\n".join(lines)


def format_full_stats(result: dict, territory: str, year: int,
                      analysis: Optional[AnalysisConfig] = None,
                      stats: Optional[StatsConfig] = None) -> str:
    """Полная ASCII-сводка по результату одного прогона process_year."""
    analysis = analysis or AnalysisConfig()
    stats = stats or StatsConfig()
    show = stats.show
    W = stats.format["separator_width"]
    sep, sub = "=" * W, "-" * W
    lines = [sep, f"  ПРОГОН: {territory}, {year}", sep]

    if show.get("composite_info"):
        lines += ["  Композит:", f"    Снимков:  {result.get('n_scenes', '—')}"]
        if result.get("used_year") and result["used_year"] != year:
            lines.append(f"    Fallback: использован {result['used_year']}")
        lines.append(sub)

    if show.get("training_info"):
        lines += [
            "  Обучение:",
            f"    Train: {result.get('train_size', '—')}",
            f"    Test:  {result.get('test_size', '—')}",
            f"    RF:    {analysis.rf_trees} деревьев, "
            f"minLeaf={analysis.rf_min_leaf}, bag={analysis.rf_bag_fraction}",
            sub,
        ]

    if show.get("accuracy"):
        lines += [format_accuracy(result["oa"], result["kappa"], stats), sub]

    if show.get("confusion_matrix") and "confusion_matrix" in result:
        lines += ["  Матрица ошибок:",
                  format_confusion_matrix(result["confusion_matrix"],
                                          analysis.classes, stats), sub]

    if show.get("per_class_metrics") and "confusion_matrix" in result:
        metrics = compute_per_class_metrics(result["confusion_matrix"], analysis.classes)
        lines += ["  Метрики по классам:", format_per_class_metrics(metrics, stats), sub]

    if show.get("areas"):
        lines += ["  Площади:",
                  format_areas_table(result["areas"], result["total_area"],
                                     analysis, stats), sub]

    if show.get("timing") and "elapsed_sec" in result:
        lines += [f"  Время: {result['elapsed_sec']:.1f} с", sep]

    return "\n".join(lines)


# ---------------------------------------------------------------------
# История прогонов и накопительная таблица
# ---------------------------------------------------------------------
def load_run_history(path: Path | str) -> list:
    """Загружает историю прогонов из JSON (пустой список, если файла нет)."""
    path = Path(path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Не удалось прочитать историю: {exc}")
    return []


def save_run_history(history: list, path: Path | str) -> None:
    """Сохраняет историю прогонов в JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def append_to_history(territory: str, year: int, result: dict,
                      history_path: Path | str,
                      cumulative_path: Path | str) -> dict:
    """Добавляет запись о прогоне в историю и накопительную CSV."""
    history = load_run_history(history_path)
    areas = result.get("areas", {})
    record = {
        "territory": territory,
        "year": year,
        "oa": result.get("oa"),
        "kappa": result.get("kappa"),
        "total_area": result.get("total_area"),
        "train_size": result.get("train_size"),
        "test_size": result.get("test_size"),
        "n_scenes": result.get("n_scenes"),
        "elapsed_sec": result.get("elapsed_sec"),
        "areas": {str(k): v for k, v in areas.items()},
        "timestamp": datetime.now().isoformat(),
    }
    history.append(record)
    save_run_history(history, history_path)
    update_cumulative_csv(record, cumulative_path)
    return record


def update_cumulative_csv(record: dict, path: Path | str) -> None:
    """Обновляет накопительную CSV-таблицу одной записью."""
    import pandas as pd

    path = Path(path)
    areas = record.get("areas", {})
    row = {
        "territory": record["territory"],
        "year": record["year"],
        "OA": record.get("oa"),
        "Kappa": record.get("kappa"),
        "Dense_km2": areas.get("0", 0),
        "Sparse_km2": areas.get("1", 0),
        "BuiltUp_km2": areas.get("2", 0),
        "Water_km2": areas.get("3", 0),
        "n_scenes": record.get("n_scenes"),
        "elapsed_sec": record.get("elapsed_sec"),
    }
    if path.exists():
        df = pd.read_csv(path)
        df = df[~((df["territory"] == row["territory"]) & (df["year"] == row["year"]))]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.sort_values(["territory", "year"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def show_cumulative_table(path: Path | str) -> Optional[object]:
    """Печатает накопительную таблицу; возвращает DataFrame или None."""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        logger.info("Накопительная таблица пока пуста.")
        return None
    df = pd.read_csv(path)
    logger.info("=" * 70)
    logger.info("НАКОПИТЕЛЬНАЯ ТАБЛИЦА ПРОГОНОВ")
    logger.info("=" * 70)
    logger.info("\n" + df.to_string(index=False))
    return df
