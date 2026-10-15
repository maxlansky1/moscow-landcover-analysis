"""Сводка метрик H2: текстовые таблицы, вердикт и сборка результатов."""
from __future__ import annotations

import numpy as np
from loguru import logger

from .._plotting import have_matplotlib
from ..config import TERRITORIES, AnalysisConfig, Config, StatsConfig
from ..pipeline import build_run_plan
from ..stats import compute_per_class_metrics, get_accuracy_mark
from ._base import _normalize_results
from .cumulative import _update_cumulative
from .plots import plot_confusion_heatmaps, plot_oa_bars, plot_oa_trend
from .verdict import _verdict


def _pick_mark(value: float, marks: list) -> str:
    """Первая метка, порог которой не превышает значение."""
    for threshold, mark in marks:
        if value >= threshold:
            return mark
    return marks[-1][1]


def summary(cfg: Config, results_by_year: dict, verbose: bool = True,
            plots: bool = True) -> dict:
    """H2: строит полную сводку метрик, графики и вердикт.

    Возвращает словарь с собранными объектами (all_importances, issues,
    successes, trend_slopes).
    """
    analysis = cfg.analysis
    stats = cfg.stats
    paths = cfg.paths
    map_cfg = cfg.map
    results_by_year = _normalize_results(results_by_year)

    logger.info("=" * 70)
    logger.info("H2: ПОЛНАЯ СВОДКА МЕТРИК")
    logger.info("=" * 70)

    _log_metrics_table(results_by_year, analysis, stats, verbose)
    _log_per_class_table(results_by_year, analysis, stats, verbose)
    _log_separability_table(results_by_year, analysis, stats, verbose)
    all_importances, mean_imp = _log_feature_importance(
        results_by_year, analysis, verbose
    )

    if plots and have_matplotlib():
        plot_oa_bars(results_by_year, analysis, map_cfg,
                     paths["PLOTS"] / "oa_by_year.png", verbose)
        plot_oa_trend(results_by_year, analysis, map_cfg,
                      paths["PLOTS"] / "oa_trend.png", verbose)
        plot_confusion_heatmaps(results_by_year, analysis, map_cfg,
                                paths["PLOTS"], verbose)
    elif plots:
        logger.warning("matplotlib недоступен — графики H2 пропущены.")

    issues, successes, trend_slopes = _verdict(
        results_by_year, analysis, stats, all_importances, mean_imp, verbose
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


def _log_metrics_table(results_by_year, analysis: AnalysisConfig,
                       stats: StatsConfig, verbose) -> None:
    if not verbose:
        return
    logger.info("=" * 70)
    logger.info("ТАБЛИЦА 1: МЕТРИКИ ПО ГОДАМ")
    logger.info("=" * 70)
    for territory_name in TERRITORIES:
        subset = results_by_year.get(territory_name, {})
        if not subset:
            logger.warning(f"[{territory_name}] Нет результатов")
            continue
        logger.info(f"\n[{territory_name}]")
        logger.info(f"  {'Год':<6}{'OA':>10}{'Kappa':>11}"
                    f"{'Dense':>8}{'Sparse':>8}{'BuiltUp':>8}{'Water':>8}")
        logger.info(f"  {'-' * 68}")
        for year in sorted(subset):
            r = subset[year]
            oa, kappa, a = r["oa"], r["kappa"], r["areas"]
            oa_str = f"{oa:>8.4f} {get_accuracy_mark(oa, 'oa', stats)}"
            kappa_str = f"{kappa:>8.4f} {get_accuracy_mark(kappa, 'kappa', stats)}"
            logger.info(f"  {year:<6}{oa_str}{kappa_str}"
                        f"{a.get(0, 0):>8.1f}{a.get(1, 0):>8.1f}"
                        f"{a.get(2, 0):>8.1f}{a.get(3, 0):>8.1f}")
        oa_mean = np.mean([subset[y]["oa"] for y in subset])
        logger.info(f"  {'-' * 68}")
        logger.info(f"  Среднее OA = {oa_mean:.4f}")


def _log_per_class_table(results_by_year, analysis: AnalysisConfig,
                         stats: StatsConfig, verbose) -> None:
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
                logger.info(f"    {cls_name:<28}{m['precision']:>12.3f}"
                            f"{m['recall']:>12.3f}{m['f1']:>11.3f} "
                            f"{_pick_mark(m['f1'], stats.f1_marks)}")


def _log_separability_table(results_by_year, analysis: AnalysisConfig,
                            stats: StatsConfig, verbose) -> None:
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
            verdict = _pick_mark(avg, stats.f1_pair_marks)
            logger.info(f"  {year:<6}{f1_dense:>12.3f}{f1_sparse:>12.3f}"
                        f"{avg:>12.3f}  {verdict}")


def _log_feature_importance(results_by_year, analysis: AnalysisConfig, verbose):
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
