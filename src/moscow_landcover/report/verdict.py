"""Вердикт H2: успехи и проблемы текущего прогона."""
from __future__ import annotations

import numpy as np
from loguru import logger

from ..boundaries import EXPECTED_AREAS
from ..config import TERRITORIES, AnalysisConfig, StatsConfig
from ..pipeline import build_run_plan
from ..stats import compute_per_class_metrics


def _verdict(results_by_year, analysis: AnalysisConfig, stats: StatsConfig,
             all_importances, mean_imp, verbose):
    issues: list = []
    successes: list = []
    trend_slopes: dict = {}

    run_plan = build_run_plan(analysis)
    n_total = len(run_plan)
    n_success = sum(len(v) for v in results_by_year.values())
    if n_success == n_total:
        successes.append(f"Все {n_total} прогонов успешны")
    else:
        issues.append(f"Провалов: {n_total - n_success}")

    for territory_name in TERRITORIES:
        for year, r in results_by_year.get(territory_name, {}).items():
            boundary_area = EXPECTED_AREAS[territory_name]
            area_diff_pct = abs(r["total_area"] - boundary_area) / boundary_area * 100
            if area_diff_pct > 2:
                issues.append(f"[{territory_name} {year}] площадь "
                              f"Δ={area_diff_pct:.1f}%")
    if not any("площадь" in i for i in issues):
        successes.append("Площади согласованы с границами во всех итерациях")

    if mean_imp:
        sorted_imp = sorted(mean_imp.items(), key=lambda x: x[1], reverse=True)
        top5_names = [f for f, _ in sorted_imp[:5]]
        texture_in_top5 = [f for f in top5_names if f in analysis.texture_bands]
        if texture_in_top5:
            successes.append(f"Текстурные признаки в топ-5: {texture_in_top5}")
        else:
            issues.append("Текстурные признаки не в топ-5 (не работают?)")

    low_f1_classes = set()
    for territory_name in TERRITORIES:
        for year, r in results_by_year.get(territory_name, {}).items():
            metrics = compute_per_class_metrics(r["confusion_matrix"],
                                                analysis.classes)
            for cls_name, m in metrics.items():
                if m["f1"] < stats.f1_min_ok:
                    low_f1_classes.add(f"{territory_name} {year} {cls_name}")
    if low_f1_classes:
        issues.append(f"Низкий F1 (<{stats.f1_min_ok:g}) у "
                      f"{len(low_f1_classes)} классов")
    else:
        successes.append(f"Все классы имеют F1 ≥ {stats.f1_min_ok:g}")

    for territory_name in TERRITORIES:
        subset = results_by_year.get(territory_name, {})
        if len(subset) >= 3:
            years = sorted(subset)
            oa_vals = [subset[y]["oa"] for y in years]
            slope = np.polyfit(years, oa_vals, 1)[0]
            trend_slopes[territory_name] = float(slope)
            if slope > 0.001:
                successes.append(f"[{territory_name}] OA растёт: {slope:+.5f}/год")
            elif slope < -0.001:
                issues.append(f"[{territory_name}] OA падает: {slope:+.5f}/год")

    if verbose:
        for s in successes:
            logger.success(f"  ✅ {s}")
        if issues:
            for issue in issues:
                logger.warning(f"  ⚠️  {issue}")
        else:
            logger.success("  Проблем не обнаружено")

    return issues, successes, trend_slopes
