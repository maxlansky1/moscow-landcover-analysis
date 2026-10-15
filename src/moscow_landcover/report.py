"""Итоговая сводка метрик и анализ (бывшая H2).

Детерминированные таблицы/вердикт живут здесь; переиспользует форматтеры
и историю прогонов из `stats`. Работает как с «живыми» результатами
`pipeline.run`, так и с JSON-загруженными (`classified` — строка).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from .boundaries import EXPECTED_AREAS
from .config import AnalysisConfig, Config
from .pipeline import build_run_plan
from .stats import append_to_history, compute_per_class_metrics, show_cumulative_table

# Значения прошлого прогона (11 признаков) — база для Δ.
BASELINE_11 = {
    "Historical": {
        2000: {"oa": 0.7130, "kappa": 0.6172},
        2007: {"oa": 0.7299, "kappa": 0.6238},
        2013: {"oa": 0.7346, "kappa": 0.6482},
        2019: {"oa": 0.7402, "kappa": 0.6536},
        2026: {"oa": 0.7163, "kappa": 0.6252},
    },
    "Greater": {
        2013: {"oa": 0.8032, "kappa": 0.7191},
        2019: {"oa": 0.8021, "kappa": 0.7361},
        2026: {"oa": 0.7963, "kappa": 0.7285},
    },
}

TERRITORIES = ("Historical", "Greater")
SHORT_CLASSES = ["Dense", "Sparse", "Built", "Water"]


def _have_matplotlib() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


def _normalize_results(results_by_year: dict) -> dict:
    """Приводит ключи areas к int (после загрузки из JSON они — строки)."""
    normalized: dict = {}
    for territory, subset in (results_by_year or {}).items():
        normalized[territory] = {}
        for year, result in (subset or {}).items():
            result = dict(result)
            areas = result.get("areas") or {}
            result["areas"] = {int(k): v for k, v in areas.items()}
            fi = result.get("feature_importance")
            if fi:
                result["feature_importance"] = {str(k): v for k, v in fi.items()}
            normalized[territory][int(year)] = result
    return normalized


def summary(cfg: Config, results_by_year: dict, verbose: bool = True,
            plots: bool = True) -> dict:
    """H2: строит полную сводку метрик, графики и вердикт.

    Возвращает словарь с собранными объектами (all_importances, issues,
    successes, trend_slopes).
    """
    analysis = cfg.analysis
    paths = cfg.paths
    plot_style = cfg.map.plot_style
    results_by_year = _normalize_results(results_by_year)

    logger.info("=" * 70)
    logger.info("H2: ПОЛНАЯ СВОДКА МЕТРИК")
    logger.info("=" * 70)

    _log_metrics_table(results_by_year, verbose)
    _log_improvements(results_by_year, verbose)
    _log_per_class_table(results_by_year, analysis, verbose)
    _log_separability_table(results_by_year, analysis, verbose)
    all_importances, mean_imp = _log_feature_importance(
        results_by_year, analysis, verbose
    )

    if plots and _have_matplotlib():
        _plot_oa_bars(results_by_year, analysis, plot_style,
                      paths["OUTPUT"] / "h2_oa_bars.png", verbose)
        _plot_oa_trend(results_by_year, analysis, plot_style,
                       paths["OUTPUT"] / "h2_oa_trend.png", verbose)
        _plot_confusion_heatmaps(results_by_year, analysis,
                                 paths["OUTPUT"], verbose)
    elif plots:
        logger.warning("matplotlib недоступен — графики H2 пропущены.")

    issues, successes, trend_slopes = _verdict(
        results_by_year, analysis, all_importances, mean_imp, verbose
    )
    _update_cumulative(results_by_year, paths, verbose)

    logger.info("=" * 70)
    if not issues:
        logger.success("✅ H2 ПРОЙДЕН: пайплайн в отличном состоянии")
        logger.success("ГОТОВО К БЛОКУ I (карты, Excel, отчёт)")
    else:
        logger.warning(f"⚠️  H2: {len(issues)} предупреждений")
    logger.info("=" * 70)

    return {
        "all_importances": all_importances,
        "mean_importance": mean_imp,
        "issues": issues,
        "successes": successes,
        "trend_slopes": trend_slopes,
    }


def _log_metrics_table(results_by_year, verbose) -> None:
    if not verbose:
        return
    logger.info("=" * 70)
    logger.info("ТАБЛИЦА 1: МЕТРИКИ С СРАВНЕНИЕМ С BASELINE (11 признаков)")
    logger.info("=" * 70)
    for territory_name in TERRITORIES:
        subset = results_by_year.get(territory_name, {})
        if not subset:
            logger.warning(f"[{territory_name}] Нет результатов")
            continue
        logger.info(f"\n[{territory_name}]")
        logger.info(f"  {'Год':<6}{'OA':>9}{'ΔOA':>9}{'Kappa':>9}{'ΔKappa':>9}"
                    f"{'Dense':>8}{'Sparse':>8}{'BuiltUp':>8}{'Water':>8}")
        logger.info(f"  {'-' * 76}")
        for year in sorted(subset):
            r = subset[year]
            oa, kappa, a = r["oa"], r["kappa"], r["areas"]
            baseline = BASELINE_11.get(territory_name, {}).get(year)
            if baseline:
                delta_oa = oa - baseline["oa"]
                delta_kappa = kappa - baseline["kappa"]
                oa_marker = ("✅" if delta_oa > 0.01
                             else "⚠️" if delta_oa < -0.01 else "—")
                oa_str = f"{oa:>8.4f}"
                delta_oa_str = f"{delta_oa:>+8.4f}{oa_marker}"
                delta_kappa_str = f"{delta_kappa:>+9.4f}"
            else:
                oa_str = (f"{oa:>8.4f} 🟢" if oa >= 0.85
                          else f"{oa:>8.4f} 🟡" if oa >= 0.75
                          else f"{oa:>8.4f} 🔴")
                delta_oa_str = f"{'—':>9}"
                delta_kappa_str = f"{'—':>9}"
            logger.info(f"  {year:<6}{oa_str}{delta_oa_str}{kappa:>9.4f}"
                        f"{delta_kappa_str}{a.get(0, 0):>8.1f}{a.get(1, 0):>8.1f}"
                        f"{a.get(2, 0):>8.1f}{a.get(3, 0):>8.1f}")
        oa_mean = np.mean([subset[y]["oa"] for y in subset])
        logger.info(f"  {'-' * 76}")
        logger.info(f"  Среднее OA = {oa_mean:.4f}")


def _log_improvements(results_by_year, verbose) -> None:
    if not verbose:
        return
    logger.info("СВОДКА УЛУЧШЕНИЙ")
    for territory_name in TERRITORIES:
        subset = results_by_year.get(territory_name, {})
        deltas = [
            r["oa"] - BASELINE_11[territory_name][year]["oa"]
            for year, r in subset.items()
            if year in BASELINE_11.get(territory_name, {})
        ]
        if not deltas:
            continue
        mean_delta, max_delta, min_delta = (np.mean(deltas), np.max(deltas),
                                            np.min(deltas))
        if mean_delta > 0.02:
            logger.success(f"[{territory_name}] ✅ Средний прирост OA: "
                           f"{mean_delta:+.4f} (макс +{max_delta:.4f}, "
                           f"мин {min_delta:+.4f})")
        elif mean_delta > 0:
            logger.info(f"[{territory_name}] Прирост OA: {mean_delta:+.4f}")
        else:
            logger.warning(f"[{territory_name}] ⚠️  Средний OA упал: "
                           f"{mean_delta:+.4f}")


def _log_per_class_table(results_by_year, analysis, verbose) -> None:
    if not verbose:
        return
    logger.info("=" * 70)
    logger.info("ТАБЛИЦА 2: PER-CLASS МЕТРИКИ")
    logger.info("=" * 70)
    for territory_name in TERRITORIES:
        subset = results_by_year.get(territory_name, {})
        if not subset:
            continue
        logger.info(f"\n[{territory_name}]")
        for year in sorted(subset):
            metrics = compute_per_class_metrics(
                subset[year]["confusion_matrix"], analysis.classes
            )
            logger.info(f"\n  {year}:")
            logger.info(f"    {'Класс':<28}{'Precision':>12}{'Recall':>12}{'F1':>12}")
            logger.info(f"    {'-' * 64}")
            for cls_name in analysis.classes:
                m = metrics[cls_name]
                f1_marker = ("✅" if m["f1"] >= 0.80
                             else "⚠️" if m["f1"] >= 0.60 else "❌")
                logger.info(f"    {cls_name:<28}{m['precision']:>12.3f}"
                            f"{m['recall']:>12.3f}{m['f1']:>11.3f} {f1_marker}")


def _log_separability_table(results_by_year, analysis, verbose) -> None:
    if not verbose:
        return
    logger.info("=" * 70)
    logger.info("ТАБЛИЦА 3: SEPARABILITY Dense ↔ Sparse (главная проблема)")
    logger.info("=" * 70)
    for territory_name in TERRITORIES:
        subset = results_by_year.get(territory_name, {})
        if not subset:
            continue
        logger.info(f"\n[{territory_name}]")
        logger.info(f"  {'Год':<6}{'F1(Dense)':>12}{'F1(Sparse)':>12}"
                    f"{'Среднее':>12}  {'вердикт'}")
        logger.info(f"  {'-' * 62}")
        for year in sorted(subset):
            metrics = compute_per_class_metrics(
                subset[year]["confusion_matrix"], analysis.classes
            )
            f1_dense = metrics["Dense Vegetation"]["f1"]
            f1_sparse = metrics["Sparse Vegetation"]["f1"]
            avg = (f1_dense + f1_sparse) / 2
            verdict = ("✅ отлично" if avg >= 0.80
                       else "⚠️ приемлемо" if avg >= 0.65 else "❌ плохо")
            logger.info(f"  {year:<6}{f1_dense:>12.3f}{f1_sparse:>12.3f}"
                        f"{avg:>12.3f}  {verdict}")


def _log_feature_importance(results_by_year, analysis, verbose):
    all_importances: dict = {}
    for territory_name in TERRITORIES:
        for year, r in results_by_year.get(territory_name, {}).items():
            fi = r.get("feature_importance")
            if fi:
                all_importances[(territory_name, year)] = fi

    mean_imp: dict = {}
    if verbose:
        logger.info("=" * 70)
        logger.info("FEATURE IMPORTANCE: важность признаков в Random Forest")
        logger.info("=" * 70)

    if not all_importances:
        logger.warning("Feature importance не собрана. Проверьте, что E7 обновлён.")
        return all_importances, mean_imp

    if verbose:
        for (territory_name, year), fi in sorted(all_importances.items()):
            top5 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]
            total = sum(fi.values())
            logger.info(f"\n[{territory_name} {year}]")
            logger.info(f"  {'Признак':<18}{'Важность':>12}{'Доля %':>10}")
            logger.info(f"  {'-' * 40}")
            for feat, imp in top5:
                logger.info(f"  {feat:<18}{imp:>12.4f}{imp / total * 100:>9.1f}%")

    avg_importance = {f: [] for f in analysis.features}
    for fi in all_importances.values():
        for feat in analysis.features:
            if feat in fi:
                avg_importance[feat].append(fi[feat])
    mean_imp = {f: float(np.mean(v)) for f, v in avg_importance.items() if v}
    total_mean = sum(mean_imp.values())

    if verbose and total_mean > 0:
        sorted_imp = sorted(mean_imp.items(), key=lambda x: x[1], reverse=True)
        logger.info("-" * 70)
        logger.info("СРЕДНЯЯ ВАЖНОСТЬ ПРИЗНАКОВ (по всем итерациям)")
        logger.info(f"  {'Место':<6}{'Признак':<18}{'Средняя важность':>18}{'Доля':>8}")
        logger.info(f"  {'-' * 52}")
        for rank, (feat, imp) in enumerate(sorted_imp, 1):
            if feat in analysis.spectral_bands:
                type_mark = "📷"
            elif feat in analysis.texture_bands:
                type_mark = "🧩"
            else:
                type_mark = "📊"
            marker = "⭐" if rank <= 5 else "  "
            logger.info(f"  {rank:<6}{feat:<18}{imp:>18.4f}"
                        f"{imp / total_mean * 100:>7.1f}%  {type_mark}{marker}")

        top5_names = [f for f, _ in sorted_imp[:5]]
        texture_in_top5 = [f for f in top5_names if f in analysis.texture_bands]
        if texture_in_top5:
            logger.success(f"✅ Текстурные признаки в топ-5: {texture_in_top5}")
        else:
            logger.warning("⚠️  Текстурные признаки не попали в топ-5")
            for f in analysis.texture_bands:
                if f in mean_imp:
                    rank = [name for name, _ in sorted_imp].index(f) + 1
                    logger.info(f"     {f}: место {rank} из {len(sorted_imp)}")

    return all_importances, mean_imp


def _plot_oa_bars(results_by_year, analysis, plot_style, out_path, verbose) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, territory_name in zip(axes, TERRITORIES):
        subset = results_by_year.get(territory_name, {})
        if not subset:
            ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
            ax.set_title(territory_name)
            continue
        years = sorted(subset)
        oa_new = [subset[y]["oa"] for y in years]
        oa_base = [BASELINE_11[territory_name][y]["oa"] for y in years
                   if y in BASELINE_11.get(territory_name, {})]
        color = (plot_style["historical_color"] if territory_name == "Historical"
                 else plot_style["greater_color"])
        x = np.arange(len(years))
        width = 0.38
        ax.bar(x - width / 2, oa_base, width, label="Baseline (11 признаков)",
               color=color, alpha=0.35, edgecolor="gray")
        bars_new = ax.bar(x + width / 2, oa_new, width, label="С 15 признаками",
                          color=color, alpha=0.85, edgecolor="black")
        ax.axhline(y=analysis.accuracy_threshold, color="green", linestyle="--",
                   alpha=0.6, linewidth=1.5,
                   label=f"Цель ({analysis.accuracy_threshold})")
        for bar, val in zip(bars_new, oa_new):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
        for i, (y, val) in enumerate(zip(years, oa_new)):
            baseline = BASELINE_11.get(territory_name, {}).get(y)
            if baseline:
                delta = val - baseline["oa"]
                ax.text(i, max(val, baseline["oa"]) + 0.04, f"Δ{delta:+.3f}",
                        ha="center", fontsize=8, fontweight="bold",
                        color="green" if delta > 0 else "red")
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{territory_name}: OA (baseline vs 15 признаков)", fontsize=12)
        ax.set_xlabel("Год")
        ax.set_ylabel("OA")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=plot_style["grid_alpha"], axis="y")
    fig.suptitle("Сравнение: baseline (11) vs обогащённый (15) набор признаков",
                 fontsize=13, y=1.02)
    _save(fig, out_path, verbose)


def _plot_oa_trend(results_by_year, analysis, plot_style, out_path, verbose) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, territory_name in zip(axes, TERRITORIES):
        subset = results_by_year.get(territory_name, {})
        if not subset:
            continue
        years = sorted(subset)
        oa_vals = [subset[y]["oa"] for y in years]
        color = (plot_style["historical_color"] if territory_name == "Historical"
                 else plot_style["greater_color"])
        ax.plot(years, oa_vals, marker="o", linewidth=2, markersize=8,
                color=color, label="OA")
        if len(years) >= 3:
            coef = np.polyfit(years, oa_vals, 1)
            trend = np.polyval(coef, years)
            ax.plot(years, trend, "--", linewidth=1.5, color="black", alpha=0.6,
                    label=f"Тренд: {coef[0]:+.5f}/год")
            ss_res = np.sum((np.array(oa_vals) - trend) ** 2)
            ss_tot = np.sum((np.array(oa_vals) - np.mean(oa_vals)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            ax.text(0.02, 0.98, f"R² = {r2:.3f}", transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
                    fontsize=10)
        ax.axhline(y=analysis.accuracy_threshold, color="green", linestyle=":",
                   alpha=0.6, label=f"Цель ({analysis.accuracy_threshold})")
        ax.set_title(f"{territory_name}: тренд OA по годам", fontsize=12)
        ax.set_xlabel("Год")
        ax.set_ylabel("OA")
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=plot_style["grid_alpha"])
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Тренд Overall Accuracy (целевые годы)", fontsize=13, y=1.02)
    _save(fig, out_path, verbose)


def _plot_confusion_heatmaps(results_by_year, analysis, out_dir, verbose) -> None:
    import matplotlib.pyplot as plt

    for territory_name in TERRITORIES:
        subset = results_by_year.get(territory_name, {})
        if not subset:
            continue
        years = sorted(subset)
        n_years = len(years)
        if n_years <= 3:
            fig, axes = plt.subplots(1, n_years, figsize=(5 * n_years, 5))
            if n_years == 1:
                axes = [axes]
        else:
            fig, axes = plt.subplots(1, n_years, figsize=(4.5 * n_years, 5))
        for ax, year in zip(axes, years):
            _plot_confusion_heatmap(
                subset[year]["confusion_matrix"], analysis.classes,
                f"{year}\nOA = {subset[year]['oa']:.3f}", ax,
            )
        fig.suptitle(f"{territory_name}: Confusion Matrix по годам "
                     f"(нормализовано по строкам)", fontsize=13, y=1.02)
        _save(fig, Path(out_dir) / f"h2_confusion_{territory_name}.png", verbose)


def _plot_confusion_heatmap(matrix, classes, title, ax) -> None:
    matrix = np.array(matrix)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = matrix / np.where(row_sums > 0, row_sums, 1)
    ax.imshow(matrix_norm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(SHORT_CLASSES, rotation=0, fontsize=9)
    ax.set_yticklabels(SHORT_CLASSES, fontsize=9)
    ax.set_xlabel("Предсказание", fontsize=9)
    ax.set_ylabel("Истина", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for i in range(len(classes)):
        for j in range(len(classes)):
            count = matrix[i, j]
            frac = matrix_norm[i, j]
            ax.text(j, i, f"{count:.0f}\n{frac:.2f}", ha="center", va="center",
                    color="white" if frac > 0.55 else "black", fontsize=8)


def _verdict(results_by_year, analysis, all_importances, mean_imp, verbose):
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
        subset = results_by_year.get(territory_name, {})
        deltas = [
            r["oa"] - BASELINE_11[territory_name][year]["oa"]
            for year, r in subset.items()
            if year in BASELINE_11.get(territory_name, {})
        ]
        if deltas:
            mean_delta = np.mean(deltas)
            if mean_delta >= 0.02:
                successes.append(f"[{territory_name}] ✅ OA вырос на "
                                 f"{mean_delta:+.4f} (текстурные признаки работают)")
            elif mean_delta >= 0:
                successes.append(f"[{territory_name}] OA вырос незначительно "
                                 f"({mean_delta:+.4f})")
            else:
                issues.append(f"[{territory_name}] OA упал на {mean_delta:+.4f} — "
                              f"текстурные признаки вредят")

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
                if m["f1"] < 0.60:
                    low_f1_classes.add(f"{territory_name} {year} {cls_name}")
    if low_f1_classes:
        issues.append(f"Низкий F1 (<0.60) у {len(low_f1_classes)} классов")
    else:
        successes.append("Все классы имеют F1 ≥ 0.60")

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


def _update_cumulative(results_by_year, paths, verbose) -> None:
    history_file = paths["INTERMEDIATE"] / "run_history.json"
    cumulative_file = paths["INTERMEDIATE"] / "cumulative_table.csv"
    if verbose:
        logger.info("Обновляю накопительную таблицу...")
    for territory_name in TERRITORIES:
        for year, result in sorted(results_by_year.get(territory_name, {}).items()):
            try:
                append_to_history(territory_name, year, result, history_file,
                                  cumulative_file)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"append_to_history {territory_name} {year}: {exc}")
    try:
        show_cumulative_table(cumulative_file)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Не удалось показать накопительную таблицу: {exc}")


def _save(fig, out_path, verbose) -> None:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if verbose:
            logger.info(f"График сохранён: {out_path}")
    plt.close(fig)
