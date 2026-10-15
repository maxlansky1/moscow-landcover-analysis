"""Диагностика качества данных (бывшие F1/F2).

F1 — доступность снимков Landsat за 2000–2026 по обеим территориям.
F2 — качество летних композитов и обучающих точек (единый проход по всем
годам, расширенные метрики — только для целевых лет).
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from .boundaries import load_boundaries
from .composite import create_composite
from .config import AnalysisConfig, Config
from .landsat import build_collection
from .logging import run_timestamp
from .pipeline import json_safe

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - scipy опционален
    scipy_stats = None
    SCIPY_AVAILABLE = False


SAMPLE_POINTS_PER_CLASS = 300
VALIDATION_SEED = 42
ANOMALY_THRESHOLD = 0.1
TERRITORIES = ("Historical", "Greater")


def _have_matplotlib() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


def _finalize(fig, out_path: Optional[Path] = None, verbose: bool = True) -> None:
    """Строит tight_layout, сохраняет и закрывает фигуру (если есть mpl)."""
    import matplotlib.pyplot as plt

    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if verbose:
            logger.info(f"График сохранён: {out_path}")
    plt.close(fig)


# =====================================================================
# F1: доступность снимков
# =====================================================================
def get_all_scenes_metadata(boundary, year_range, season_months=(6, 7, 8),
                            analysis: Optional[AnalysisConfig] = None) -> list:
    """Метаданные всех летних снимков в границах за один запрос к GEE."""
    import ee

    analysis = analysis or AnalysisConfig()
    sat_ids = list(analysis.satellites_early) + list(analysis.satellites_late)
    col = build_collection(sat_ids)

    start = f"{year_range[0]}-06-01"
    end = f"{year_range[-1]}-09-01"
    col = col.filterDate(start, end).filterBounds(boundary)

    def make_feature(img):
        date = ee.Date(img.get("DATE_ACQUIRED"))
        return ee.Feature(None, {
            "year": date.get("year"),
            "month": date.get("month"),
            "cloud": img.get("CLOUD_COVER"),
            "sensor": img.get("SPACECRAFT_ID"),
        })

    features = col.map(make_feature)
    features = features.filter(ee.Filter.inList("month", list(season_months)))

    info = features.getInfo()
    return [f["properties"] for f in info.get("features", [])]


def build_scene_summary(metadata_list: list) -> pd.DataFrame:
    """Агрегирует метаданные снимков по годам."""
    columns = ["year", "n_scenes", "median_cloud", "min_cloud",
               "max_cloud", "sensors"]
    if not metadata_list:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(metadata_list)
    df["year"] = df["year"].astype(int)
    df["cloud"] = df["cloud"].astype(float)

    def agg_sensors(series):
        return ",".join(sorted(set(series.dropna().astype(str))))

    return df.groupby("year").agg(
        n_scenes=("cloud", "count"),
        median_cloud=("cloud", "median"),
        min_cloud=("cloud", "min"),
        max_cloud=("cloud", "max"),
        sensors=("sensor", agg_sensors),
    ).reset_index()


def availability(cfg: Config, years=None, verbose: bool = True,
                 plots: bool = True) -> dict:
    """F1: проверяет доступность снимков по годам для обеих территорий.

    Возвращает {territory: DataFrame} и сохраняет diagnostics_f1_scenes.json.
    """
    analysis = cfg.analysis
    paths = cfg.paths
    min_images = analysis.min_images_for_composite
    all_years = list(years) if years is not None else list(
        range(analysis.historical_start, analysis.historical_end + 1)
    )

    logger.info("=" * 70)
    logger.info("F1: ДОСТУПНОСТЬ СНИМКОВ 2000-2026")
    logger.info("=" * 70)

    boundaries, _ = load_boundaries(paths)

    f1_data: dict = {}
    for territory_name in TERRITORIES:
        logger.info(f"[F1] Собираю метаданные снимков: {territory_name}...")
        t0 = time.time()

        metadata = get_all_scenes_metadata(boundaries[territory_name], all_years,
                                           analysis=analysis)
        summary = build_scene_summary(metadata)

        summary = pd.DataFrame({"year": all_years}).merge(
            summary, on="year", how="left"
        )
        summary["n_scenes"] = summary["n_scenes"].fillna(0).astype(int)
        summary["sensors"] = summary["sensors"].fillna("")
        summary["needs_fallback"] = summary["n_scenes"] < min_images

        f1_data[territory_name] = summary
        logger.info(f"[F1] {territory_name}: собрано за {time.time()-t0:.1f} с, "
                    f"всего снимков за период: {summary['n_scenes'].sum()}")

    _log_availability_tables(f1_data, analysis, min_images, verbose)
    _log_fallback_analysis(f1_data, analysis, min_images, verbose)

    if plots and _have_matplotlib():
        path = paths["INTERMEDIATE"] / "diagnostics_f1_scenes.png"
        _plot_availability(f1_data, min_images, path, verbose)
    elif plots:
        logger.warning("matplotlib недоступен — графики F1 пропущены.")

    f1_json = {
        "run_timestamp": run_timestamp(),
        "years": all_years,
        "min_images_for_composite": min_images,
        "data": {t: json_safe(df.to_dict(orient="records"))
                 for t, df in f1_data.items()},
    }
    f1_file = paths["INTERMEDIATE"] / "diagnostics_f1_scenes.json"
    f1_file.parent.mkdir(parents=True, exist_ok=True)
    with open(f1_file, "w", encoding="utf-8") as f:
        json.dump(f1_json, f, indent=2, ensure_ascii=False)

    logger.success(f"Результаты F1 сохранены: {f1_file.name}")
    logger.success("✅ F1 ЗАВЕРШЁН: доступность снимков проверена")
    return f1_data


def _log_availability_tables(f1_data, analysis, min_images, verbose) -> None:
    if not verbose:
        return
    logger.info("-" * 70)
    logger.info("СВОДКА ПО ГОДАМ (все 2000-2026)")
    for territory_name, df in f1_data.items():
        logger.info(f"\n[{territory_name}]")
        logger.info(f"  {'Год':<6}{'Снимков':>10}{'Медиана':>10}{'Мин':>8}"
                    f"{'Макс':>8}  {'Fallback':<10}{'Сенсоры'}")
        logger.info(f"  {'-' * 80}")
        for _, row in df.iterrows():
            med, mn, mx = row["median_cloud"], row["min_cloud"], row["max_cloud"]
            fallback = "⚠️ ДА" if row["needs_fallback"] else "нет"
            marker = ""
            if int(row["n_scenes"]) == 0:
                marker = " ❌"
            elif row["needs_fallback"]:
                marker = " ⚠️"
            logger.info(
                f"  {int(row['year']):<6}{int(row['n_scenes']):>10}"
                f"{(f'{med:.1f}' if pd.notna(med) else '—'):>10}"
                f"{(f'{mn:.1f}' if pd.notna(mn) else '—'):>8}"
                f"{(f'{mx:.1f}' if pd.notna(mx) else '—'):>8}  "
                f"{fallback:<10}{row['sensors']}{marker}"
            )


def _log_fallback_analysis(f1_data, analysis, min_images, verbose) -> None:
    if not verbose:
        return
    logger.info("-" * 70)
    logger.info("АНАЛИЗ FALLBACK")
    for territory_name, df in f1_data.items():
        problem_years = df[df["needs_fallback"]]["year"].astype(int).tolist()
        zero_years = df[df["n_scenes"] == 0]["year"].astype(int).tolist()
        if not problem_years:
            logger.success(f"[{territory_name}] ✅ Все годы имеют "
                           f"≥{min_images} снимков")
        else:
            logger.warning(f"[{territory_name}] ⚠️  Годы с недостатком снимков "
                           f"(<{min_images}): {problem_years}")
            if zero_years:
                logger.error(f"[{territory_name}] ❌ Годы БЕЗ снимков совсем: "
                             f"{zero_years}")

    logger.info("-" * 70)
    logger.info("ПРОВЕРКА ЦЕЛЕВЫХ ГОДОВ")
    for territory_name, df in f1_data.items():
        target_years = (analysis.historical_years if territory_name == "Historical"
                        else analysis.greater_years)
        logger.info(f"\n[{territory_name}] целевые годы: {target_years}")
        for year in target_years:
            row = df[df["year"] == year]
            if row.empty:
                logger.error(f"  {year}: ❌ нет данных")
                continue
            row = row.iloc[0]
            n = int(row["n_scenes"])
            med = row["median_cloud"]
            med_str = f"{med:.1f}%" if pd.notna(med) else "—"
            if n >= min_images:
                logger.success(f"  {year}: ✅ {n} снимков, медиана облачности "
                               f"{med_str}")
            elif n > 0:
                logger.warning(f"  {year}: ⚠️  только {n} снимков "
                               f"(нужно {min_images}), будет fallback")
            else:
                logger.error(f"  {year}: ❌ нет снимков")


def _plot_availability(f1_data, min_images, out_path, verbose) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

    ax1 = axes[0]
    for territory_name, df in f1_data.items():
        color = "steelblue" if territory_name == "Historical" else "darkorange"
        ax1.bar(df["year"] + (0.2 if territory_name == "Greater" else -0.2),
                df["n_scenes"], width=0.4, alpha=0.75,
                label=territory_name, color=color)
    ax1.axhline(y=min_images, color="red", linestyle="--", alpha=0.7,
                label=f"Минимум ({min_images})")
    ax1.set_ylabel("Количество снимков", fontsize=12)
    ax1.set_title("Доступность снимков Landsat за лето (2000–2026)", fontsize=13)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for territory_name, df in f1_data.items():
        color = "steelblue" if territory_name == "Historical" else "darkorange"
        valid_df = df.dropna(subset=["median_cloud"])
        ax2.plot(valid_df["year"], valid_df["median_cloud"], marker="o",
                 linewidth=2, markersize=5, label=territory_name, color=color)
    ax2.axhline(y=20, color="red", linestyle="--", alpha=0.7,
                label="20% (ориентир)")
    ax2.set_xlabel("Год", fontsize=12)
    ax2.set_ylabel("Медиана облачности (%)", fontsize=12)
    ax2.set_title("Медиана облачности летних снимков", fontsize=13)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(2000, 2027, 2))
    ax2.tick_params(axis="x", rotation=45)

    _finalize(fig, out_path, verbose)


# =====================================================================
# F2: диагностика композитов и обучающих точек
# =====================================================================
def subsample_points_tiered(points, n_per_class: int, seed: int = VALIDATION_SEED,
                            n_classes: int = 4):
    """Берёт n_per_class точек из каждого класса, рандомизированно."""
    import ee

    subsets = []
    for cls_id in range(n_classes):
        cls_pts = points.filter(ee.Filter.eq("class", cls_id))
        subset = (cls_pts.randomColumn("r", seed=seed).sort("r").limit(n_per_class))
        subsets.append(subset)
    return ee.FeatureCollection(subsets).flatten()


def compute_composite_stats_light(composite, boundary, area_km2: float,
                                  analysis: Optional[AnalysisConfig] = None):
    """Базовые метрики композита: coverage, средние, cloud_residue.

    Ключевые моменты (историю правок см. в ноутбуке F2):
      • geom = boundary.geometry();
      • crs=ANALYSIS_CRS во всех reduceRegion;
      • total_pixels — тем же reduceRegion на константном изображении.
    """
    import ee

    analysis = analysis or AnalysisConfig()
    try:
        geom = boundary.geometry()

        cloud_flag = (
            composite.select("blue").gt(0.25)
            .And(composite.select("nir").lt(0.15))
            .rename("cloud_flag")
        )
        img = composite.addBands(cloud_flag)

        stats_ms = img.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(),
                                              sharedInputs=True),
            geometry=geom, scale=analysis.gee_scale, crs=analysis.analysis_crs,
            maxPixels=analysis.gee_max_pixels, tileScale=8,
        ).getInfo() or {}

        stats_count = img.select(["NDVI"]).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geom, scale=analysis.gee_scale, crs=analysis.analysis_crs,
            maxPixels=analysis.gee_max_pixels, tileScale=8,
        ).getInfo() or {}

        cloud_count = img.select("cloud_flag").reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geom, scale=analysis.gee_scale, crs=analysis.analysis_crs,
            maxPixels=analysis.gee_max_pixels, tileScale=8,
        ).getInfo().get("cloud_flag", 0) or 0

        total_pixels = ee.Image.constant(1).rename("total").reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geom, scale=analysis.gee_scale, crs=analysis.analysis_crs,
            maxPixels=analysis.gee_max_pixels, tileScale=8,
        ).getInfo().get("total", 0) or 0

        result = {}
        for band in analysis.features + ["red", "green", "blue", "nir",
                                         "swir1", "swir2"]:
            key_mean = f"{band}_mean"
            if key_mean in stats_ms:
                result[key_mean] = stats_ms[key_mean]

        result["NDVI_std"] = stats_ms.get("NDVI_stdDev")

        valid_pixels = stats_count.get("NDVI", 0) or 0
        result["valid_pixels"] = valid_pixels
        result["total_pixels"] = total_pixels
        result["coverage_pct"] = (
            valid_pixels / total_pixels * 100 if total_pixels > 0 else 0
        )
        result["cloud_residue_pct"] = (
            cloud_count / valid_pixels * 100 if valid_pixels > 0 else 0
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.error(f"    compute_composite_stats_light: {exc}")
        return None


def compute_composite_percentiles(composite, boundary,
                                  analysis: Optional[AnalysisConfig] = None):
    """NDVI percentiles (p10/p50/p90) и bimodality (только для целевых лет)."""
    import ee

    analysis = analysis or AnalysisConfig()
    try:
        geom = boundary.geometry()
        stats = composite.select("NDVI").reduceRegion(
            reducer=ee.Reducer.percentile([10, 50, 90]),
            geometry=geom, scale=analysis.gee_scale, crs=analysis.analysis_crs,
            maxPixels=analysis.gee_max_pixels, tileScale=8,
        ).getInfo()
        if stats is None:
            return None

        p10 = stats.get("NDVI_p10")
        p50 = stats.get("NDVI_p50")
        p90 = stats.get("NDVI_p90")
        result = {"NDVI_p10": p10, "NDVI_p50": p50, "NDVI_p90": p90}
        if p10 is not None and p50 is not None and p90 is not None:
            result["NDVI_bimodality"] = p50 - (p10 + p90) / 2
        else:
            result["NDVI_bimodality"] = None
        return result
    except Exception as exc:  # noqa: BLE001
        logger.error(f"    compute_composite_percentiles: {exc}")
        return None


def compute_training_validation(composite, validation_points,
                                boundary_name: str = "",
                                analysis: Optional[AnalysisConfig] = None,
                                sample_points_per_class: int = SAMPLE_POINTS_PER_CLASS):
    """Доля обучающих точек на валидных пикселях + per-class метрики NDVI.

    sampleRegions принимает `projection`, а НЕ `crs`.
    """
    import ee

    analysis = analysis or AnalysisConfig()
    tag = f"[Val:{boundary_name}]" if boundary_name else "[Val]"

    try:
        samples = composite.select(analysis.features).sampleRegions(
            collection=validation_points,
            properties=["class"],
            scale=analysis.gee_scale,
            projection=ee.Projection(analysis.analysis_crs),
            geometries=False,
            tileScale=4,
        )
        fc_info = samples.getInfo()
        features = fc_info.get("features", [])
        if not features:
            logger.warning(f"{tag} 0 валидных точек")
            return None

        df = pd.DataFrame([f["properties"] for f in features])
        df["class"] = df["class"].astype(int)

        requested_total = sample_points_per_class * analysis.n_classes
        result = {
            "valid_points_total": len(df),
            "requested_total": requested_total,
            "valid_pct": len(df) / requested_total * 100 if requested_total > 0 else 0,
            "valid_points_per_class": {},
            "NDVI_mean_per_class": {},
            "NDVI_std_per_class": {},
        }

        for cls_id in range(analysis.n_classes):
            cls_df = df[df["class"] == cls_id]
            n = len(cls_df)
            result["valid_points_per_class"][cls_id] = n
            if n > 0:
                result["NDVI_mean_per_class"][cls_id] = float(cls_df["NDVI"].mean())
                result["NDVI_std_per_class"][cls_id] = float(cls_df["NDVI"].std())
            else:
                result["NDVI_mean_per_class"][cls_id] = None
                result["NDVI_std_per_class"][cls_id] = None

        ndvi_dense = result["NDVI_mean_per_class"].get(0)
        ndvi_sparse = result["NDVI_mean_per_class"].get(1)
        if ndvi_dense is not None and ndvi_sparse is not None:
            result["separability"] = ndvi_dense - ndvi_sparse
        else:
            result["separability"] = None
        return result
    except Exception as exc:  # noqa: BLE001
        logger.error(f"{tag} Ошибка: {exc}")
        return None


def run_diagnostics(cfg: Config, availability_data=None, verbose: bool = True,
                    plots: bool = True, training_points_map=None) -> dict:
    """F2: единый проход по всем годам × 2 территориям.

    Сохраняет diagnostics_f2.json и возвращает словарь с df_f2/трендами.
    """
    import ee

    from . import points as points_module

    analysis = cfg.analysis
    paths = cfg.paths
    all_years = list(range(analysis.historical_start, analysis.historical_end + 1))
    target_years_by_territory = {
        "Historical": set(analysis.historical_years),
        "Greater": set(analysis.greater_years),
    }

    logger.info("=" * 70)
    logger.info("F2: ДИАГНОСТИКА КОМПОЗИТОВ И ОБУЧАЮЩИХ ТОЧЕК")
    logger.info("=" * 70)
    if not SCIPY_AVAILABLE:
        logger.warning("scipy не установлен — p-value не будет рассчитан")
    if availability_data is not None:
        logger.info("F2: данные F1 получены и доступны для сверки.")

    boundaries, areas = load_boundaries(paths)

    # --- обучающие точки для валидации (один раз на территорию) ---
    training_points_map = dict(training_points_map or {})
    for name in TERRITORIES:
        if name not in training_points_map:
            pts, _ = points_module.generate_training_data(
                boundary=boundaries[name], year=analysis.training_year,
                analysis=analysis, boundary_name=name, verbose=verbose,
            )
            training_points_map[name] = pts

    validation_points = {
        name: subsample_points_tiered(
            training_points_map[name], SAMPLE_POINTS_PER_CLASS,
            VALIDATION_SEED, analysis.n_classes,
        )
        for name in TERRITORIES
        if training_points_map.get(name) is not None
    }

    sat_ids = list(analysis.satellites_early) + list(analysis.satellites_late)
    collection_all = build_collection(sat_ids)

    f2_data = {name: [] for name in TERRITORIES}
    f2_failed: list = []
    total_iterations = len(all_years) * len(TERRITORIES)
    iteration = 0
    t_start = time.time()

    for territory_name in TERRITORIES:
        boundary = boundaries[territory_name]
        area_km2 = areas[territory_name]
        target_years = target_years_by_territory[territory_name]
        points = validation_points.get(territory_name)
        logger.info(f"[F2:{territory_name}] Целевые годы: {sorted(target_years)}")

        for year in all_years:
            iteration += 1
            is_target = year in target_years
            t0 = time.time()
            tag = "FULL" if is_target else "light"

            composite, comp_info = create_composite(
                collection=collection_all, boundary=boundary, year=year,
                territory_name=territory_name, analysis=analysis, verbose=False,
            )
            if composite is None:
                logger.error(f"  [{iteration:>2}/{total_iterations}] "
                             f"{territory_name} {year}: композит не создан")
                f2_failed.append({"territory": territory_name, "year": year,
                                  "reason": "composite_failed"})
                continue

            stats = compute_composite_stats_light(composite, boundary, area_km2,
                                                  analysis)
            if stats is None:
                logger.error(f"  [{iteration:>2}/{total_iterations}] "
                             f"{territory_name} {year}: reduceRegion failed")
                f2_failed.append({"territory": territory_name, "year": year,
                                  "reason": "reduce_failed"})
                continue

            if is_target:
                perc = compute_composite_percentiles(composite, boundary, analysis)
                if perc:
                    stats.update(perc)
                if points is not None:
                    val = compute_training_validation(composite, points,
                                                      territory_name, analysis)
                    if val:
                        stats["training_validation"] = val

            stats["year"] = year
            stats["territory"] = territory_name
            stats["n_scenes"] = comp_info["n_scenes"]
            stats["used_year"] = comp_info["used_year"]
            stats["fallback_used"] = comp_info["fallback_used"]
            stats["is_target"] = is_target
            f2_data[territory_name].append(stats)

            elapsed = time.time() - t0
            ndvi_m = stats.get("NDVI_mean", 0) or 0
            cov = stats.get("coverage_pct", 0)
            extra = ""
            if is_target and "training_validation" in stats:
                v = stats["training_validation"]
                extra = f", val={v['valid_pct']:.0f}%"
            logger.info(
                f"  [{iteration:>2}/{total_iterations}] {territory_name} {year} "
                f"[{tag:>5}]: n={comp_info['n_scenes']:>2}, cov={cov:>5.1f}%, "
                f"NDVI={ndvi_m:>5.3f}{extra}, {elapsed:.1f}с"
            )

    elapsed_min = (time.time() - t_start) / 60
    n_success = sum(len(v) for v in f2_data.values())
    logger.success(f"Сбор завершён за {elapsed_min:.1f} мин. "
                   f"Успешно: {n_success}/{total_iterations}")
    for item in f2_failed:
        logger.warning(f"  {item['territory']} {item['year']}: {item['reason']}")

    f2_json = {
        "run_timestamp": run_timestamp(),
        "years": all_years,
        "target_years": {k: sorted(v) for k, v in target_years_by_territory.items()},
        "sample_points_per_class": SAMPLE_POINTS_PER_CLASS,
        "failed": f2_failed,
        "data": json_safe(f2_data),
    }
    f2_file = paths["INTERMEDIATE"] / "diagnostics_f2.json"
    f2_file.parent.mkdir(parents=True, exist_ok=True)
    with open(f2_file, "w", encoding="utf-8") as f:
        json.dump(f2_json, f, indent=2, ensure_ascii=False)
    logger.success(f"Результаты F2 сохранены: {f2_file.name}")

    df_f2 = pd.DataFrame(f2_data["Historical"] + f2_data["Greater"])
    if not df_f2.empty:
        df_f2 = df_f2.sort_values(["territory", "year"]).reset_index(drop=True)

    _log_spectral_table(df_f2, analysis, verbose)
    _log_ndvi_distribution(df_f2, analysis, verbose)
    trend_results = _compute_trends(df_f2, verbose)
    _log_training_validity(df_f2, analysis, verbose)

    if plots and _have_matplotlib() and not df_f2.empty:
        _plot_ndvi_timeseries(df_f2, cfg.map.plot_style,
                              paths["INTERMEDIATE"] / "diagnostics_f2_ndvi.png",
                              verbose)
        _plot_ndvi_distribution(df_f2, cfg.map.plot_style,
                                paths["INTERMEDIATE"] / "diagnostics_f2_dist.png",
                                verbose)
        _plot_trends(df_f2, trend_results, cfg.map.plot_style,
                     paths["INTERMEDIATE"] / "diagnostics_f2_trends.png", verbose)
    elif plots:
        logger.warning("matplotlib недоступен — графики F2 пропущены.")

    issues = _log_verdict(df_f2, analysis, verbose)
    logger.success("✅ F2 ЗАВЕРШЁН")
    return {"df_f2": df_f2, "trend_results": trend_results,
            "failed": f2_failed, "issues": issues, "elapsed_min": elapsed_min}


def _log_spectral_table(df_f2, analysis, verbose) -> None:
    if not verbose or df_f2.empty:
        return
    logger.info("=" * 70)
    logger.info("ТАБЛИЦА 1: СПЕКТРАЛЬНАЯ СВОДКА (все годы)")
    logger.info("=" * 70)
    for territory_name in TERRITORIES:
        logger.info(f"\n[{territory_name}]")
        logger.info(f"  {'Год':<6}{'n':>4}{'cov%':>7}{'NDVI_m':>8}{'NDVI_s':>8}"
                    f"{'NDWI_m':>8}{'NDBI_m':>8}{'cloud%':>8}  flag")
        logger.info(f"  {'-' * 72}")
        subset = df_f2[df_f2["territory"] == territory_name].sort_values("year")
        for _, row in subset.iterrows():
            n = int(row["n_scenes"]) if pd.notna(row["n_scenes"]) else 0
            cov = row["coverage_pct"] if pd.notna(row["coverage_pct"]) else 0
            ndvi_m = row["NDVI_mean"] if pd.notna(row["NDVI_mean"]) else 0
            ndvi_s = row["NDVI_std"] if pd.notna(row["NDVI_std"]) else 0
            ndwi_m = row["NDWI_mean"] if pd.notna(row["NDWI_mean"]) else 0
            ndbi_m = row["NDBI_mean"] if pd.notna(row["NDBI_mean"]) else 0
            cloud = (row["cloud_residue_pct"]
                     if pd.notna(row["cloud_residue_pct"]) else 0)
            flags = []
            if cov < 90:
                flags.append("cov")
            if cloud > 1.0:
                flags.append("cloud")
            if ndvi_m < 0.3 or ndvi_m > 0.8:
                flags.append("NDVI")
            flag_str = ("⚠ " + " ".join(flags)) if flags else ""
            target_marker = " *" if row["is_target"] else "  "
            logger.info(f"  {int(row['year']):<6}{n:>4}{cov:>7.1f}{ndvi_m:>8.3f}"
                        f"{ndvi_s:>8.3f}{ndwi_m:>8.3f}{ndbi_m:>8.3f}{cloud:>8.2f}"
                        f"  {flag_str}{target_marker}")
    logger.info("\n  * — целевой год (полная диагностика)")


def _log_ndvi_distribution(df_f2, analysis, verbose) -> None:
    if not verbose or df_f2.empty:
        return
    logger.info("=" * 70)
    logger.info("ТАБЛИЦА 2: РАСПРЕДЕЛЕНИЕ NDVI (только целевые годы)")
    logger.info("=" * 70)
    for territory_name in TERRITORIES:
        logger.info(f"\n[{territory_name}]")
        logger.info(f"  {'Год':<6}{'p10':>8}{'p50':>8}{'p90':>8}{'bimod':>10}"
                    f"  интерпретация")
        logger.info(f"  {'-' * 65}")
        subset = df_f2[(df_f2["territory"] == territory_name)
                       & (df_f2["is_target"])].sort_values("year")
        for _, row in subset.iterrows():
            p10 = row.get("NDVI_p10")
            p50 = row.get("NDVI_p50")
            p90 = row.get("NDVI_p90")
            bimod = row.get("NDVI_bimodality")
            if pd.isna(p10) or pd.isna(p50) or pd.isna(p90):
                logger.info(f"  {int(row['year']):<6}{'—':>8}{'—':>8}{'—':>8}"
                            f"{'—':>10}")
                continue
            if bimod is None or pd.isna(bimod):
                interp = "—"
            elif abs(bimod) < 0.05:
                interp = "плоское"
            elif bimod > 0.10:
                interp = "двугорбое (норма)"
            elif bimod > 0.05:
                interp = "слабо двугорбое"
            else:
                interp = "скошено"
            logger.info(f"  {int(row['year']):<6}{p10:>8.3f}{p50:>8.3f}"
                        f"{p90:>8.3f}{bimod:>10.3f}  {interp}")


def _compute_trends(df_f2, verbose) -> dict:
    if verbose and not df_f2.empty:
        logger.info("=" * 70)
        logger.info("ТАБЛИЦА 3: ТРЕНДЫ И АНОМАЛИИ (все годы)")
        logger.info("=" * 70)

    trend_results: dict = {}
    if df_f2.empty:
        return trend_results

    for territory_name in TERRITORIES:
        subset = df_f2[df_f2["territory"] == territory_name].sort_values("year")
        subset = subset.dropna(subset=["NDVI_mean"])
        if len(subset) < 3:
            logger.warning(f"[{territory_name}] Недостаточно данных для тренда")
            continue

        years = subset["year"].values.astype(float)
        ndvi = subset["NDVI_mean"].values.astype(float)
        coef = np.polyfit(years, ndvi, 1)
        slope, intercept = coef[0], coef[1]
        trend = np.polyval(coef, years)
        residuals = ndvi - trend
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((ndvi - ndvi.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        if SCIPY_AVAILABLE:
            r, p_value = scipy_stats.pearsonr(years, ndvi)
        else:
            r, p_value = np.nan, np.nan

        anomaly_mask = np.abs(residuals) > ANOMALY_THRESHOLD
        anomaly_years = subset[anomaly_mask]["year"].astype(int).tolist()

        trend_results[territory_name] = {
            "slope_per_year": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "pearson_r": float(r) if not np.isnan(r) else None,
            "p_value": float(p_value) if not np.isnan(p_value) else None,
            "anomaly_years": anomaly_years,
        }

        if verbose:
            logger.info(f"\n[{territory_name}]")
            logger.info(f"  Линейный тренд NDVI: {slope:+.5f}/год")
            logger.info(f"  R² = {r_squared:.3f}, Pearson r = {r:.3f}, "
                        f"p = {p_value:.4f}")
            if slope > 0.001:
                logger.success(f"  ✅ Растительность растёт: +{slope*10:.3f} "
                               f"за 10 лет")
            elif slope < -0.001:
                logger.warning(f"  ⚠️  Растительность снижается: {slope*10:.3f} "
                               f"за 10 лет")
            else:
                logger.info("  Тренд слабый (|slope| < 0.001)")
            if anomaly_years:
                logger.warning(f"  Аномалии (|residual| > {ANOMALY_THRESHOLD}): "
                               f"{anomaly_years}")
            else:
                logger.success("  ✅ Аномалий нет")
    return trend_results


def _log_training_validity(df_f2, analysis, verbose) -> None:
    if not verbose or df_f2.empty:
        return
    logger.info("=" * 70)
    logger.info("ТАБЛИЦА 4: ВАЛИДНОСТЬ ОБУЧАЮЩИХ ТОЧЕК (только целевые годы)")
    logger.info("=" * 70)
    logger.info(f"Из {SAMPLE_POINTS_PER_CLASS} запрошенных точек/класс — "
                f"сколько попало на валидные пиксели.")
    for territory_name in TERRITORIES:
        logger.info(f"\n[{territory_name}]")
        subset = df_f2[(df_f2["territory"] == territory_name)
                       & (df_f2["is_target"])].sort_values("year")
        for _, row in subset.iterrows():
            val = row.get("training_validation")
            if val is None:
                logger.warning(f"  {int(row['year'])}: нет данных валидации")
                continue
            year = int(row["year"])
            logger.info(f"  {year}: {val['valid_points_total']}/"
                        f"{val['requested_total']} точек валидны "
                        f"({val['valid_pct']:.0f}%)")
            logger.info(f"    {'Класс':<28}{'Точек':>8}{'NDVI_mean':>12}"
                        f"{'NDVI_std':>10}")
            logger.info(f"    {'-' * 58}")
            for cls_id in range(analysis.n_classes):
                n = val["valid_points_per_class"].get(cls_id, 0)
                ndvi_m = val["NDVI_mean_per_class"].get(cls_id)
                ndvi_s = val["NDVI_std_per_class"].get(cls_id)
                ndvi_m_str = f"{ndvi_m:>12.3f}" if ndvi_m is not None else f"{'—':>12}"
                ndvi_s_str = f"{ndvi_s:>10.3f}" if ndvi_s is not None else f"{'—':>10}"
                logger.info(f"    {analysis.classes_ru[cls_id]:<28}{n:>8}"
                            f"{ndvi_m_str}{ndvi_s_str}")
            sep = val.get("separability")
            if sep is not None:
                sep_marker = "✅" if sep > 0.15 else "⚠️" if sep > 0.05 else "❌"
                logger.info(f"    Separability (Dense − Sparse): {sep:+.3f} "
                            f"{sep_marker}")


def _plot_ndvi_timeseries(df_f2, plot_style, out_path, verbose) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, territory_name in zip(axes, TERRITORIES):
        subset = df_f2[df_f2["territory"] == territory_name].sort_values("year")
        color = (plot_style["historical_color"] if territory_name == "Historical"
                 else plot_style["greater_color"])
        ax.plot(subset["year"], subset["NDVI_mean"], linewidth=1.2,
                color=color, alpha=0.5, zorder=2)
        for _, row in subset.iterrows():
            if pd.isna(row["coverage_pct"]):
                continue
            cov = row["coverage_pct"]
            m_color = (plot_style["ok_marker"] if cov >= 95
                       else plot_style["warn_marker"] if cov >= 90
                       else plot_style["error_marker"])
            size = 110 if row["is_target"] else 60
            edge = "black" if row["is_target"] else "none"
            lw = 1.0 if row["is_target"] else 0
            ax.scatter(row["year"], row["NDVI_mean"], color=m_color, s=size,
                       zorder=5, edgecolors=edge, linewidths=lw)

        ax2 = ax.twinx()
        ax2.plot(subset["year"], subset["NDWI_mean"], marker="s", linewidth=1.0,
                 markersize=3, alpha=0.5, color="steelblue", label="NDWI")
        ax2.plot(subset["year"], subset["NDBI_mean"], marker="^", linewidth=1.0,
                 markersize=3, alpha=0.5, color="saddlebrown", label="NDBI")
        ax2.set_ylim(-1, 1)
        ax2.set_ylabel("NDWI / NDBI", color="gray", fontsize=9)

        ax.set_title(f"{territory_name}", fontsize=12)
        ax.set_xlabel("Год")
        ax.set_ylabel("NDVI mean")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=plot_style["grid_alpha"])

        legend_elems = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="green",
                   markersize=8, label="coverage ≥95%"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="orange",
                   markersize=8, label="coverage 90–95%"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=8, label="coverage <90%"),
        ]
        ax.legend(handles=legend_elems, loc="lower left", fontsize=8)

    fig.suptitle("NDVI mean по годам (крупные маркеры = целевые годы)",
                 fontsize=13, y=1.02)
    _finalize(fig, out_path, verbose)


def _plot_ndvi_distribution(df_f2, plot_style, out_path, verbose) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, territory_name in zip(axes, TERRITORIES):
        subset = df_f2[
            (df_f2["territory"] == territory_name)
            & (df_f2["is_target"])
            & (df_f2["NDVI_p50"].notna())
        ].sort_values("year")
        color = (plot_style["historical_color"] if territory_name == "Historical"
                 else plot_style["greater_color"])
        if subset.empty:
            ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
            ax.set_title(territory_name)
            continue
        for _, row in subset.iterrows():
            year = row["year"]
            ax.plot([year, year], [row["NDVI_p10"], row["NDVI_p90"]],
                    color=color, linewidth=6, alpha=0.4, solid_capstyle="round")
            ax.scatter(year, row["NDVI_p50"], color=color, s=100, zorder=5,
                       edgecolors="black", linewidths=1)
        for _, row in subset.iterrows():
            bimod = row["NDVI_bimodality"]
            if pd.notna(bimod):
                ax.annotate(f"{bimod:.2f}", (row["year"], row["NDVI_p90"] + 0.03),
                            ha="center", fontsize=8, color="gray")
        ax.set_title(f"{territory_name}: p10–p50–p90 (аннотация = bimodality)",
                     fontsize=12)
        ax.set_xlabel("Год")
        ax.set_ylabel("NDVI")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=plot_style["grid_alpha"])
    fig.suptitle("Распределение NDVI в целевые годы", fontsize=13, y=1.02)
    _finalize(fig, out_path, verbose)


def _plot_trends(df_f2, trend_results, plot_style, out_path, verbose) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")
    for col, territory_name in enumerate(TERRITORIES):
        if territory_name not in trend_results:
            continue
        subset = df_f2[df_f2["territory"] == territory_name].sort_values("year")
        subset = subset.dropna(subset=["NDVI_mean"])
        color = (plot_style["historical_color"] if territory_name == "Historical"
                 else plot_style["greater_color"])
        years = subset["year"].values.astype(float)
        ndvi = subset["NDVI_mean"].values.astype(float)
        coef = np.polyfit(years, ndvi, 1)
        trend = np.polyval(coef, years)
        residuals = ndvi - trend

        ax = axes[0, col]
        ax.scatter(years, ndvi, s=50, color=color, zorder=5,
                   edgecolors="black", linewidths=0.5, alpha=0.8)
        target_mask = subset["is_target"].values
        if target_mask.any():
            ax.scatter(years[target_mask], ndvi[target_mask], s=140,
                       facecolors="none", edgecolors="black", linewidths=2,
                       zorder=6, label="Целевые")
        ax.plot(years, trend, "--", linewidth=plot_style["trend_linewidth"],
                color="black", alpha=0.7, label=f"Тренд: {coef[0]:+.5f}/год")
        tr = trend_results[territory_name]
        p_str = f"p = {tr['p_value']:.4f}" if tr["p_value"] is not None else "p = —"
        ax.text(0.02, 0.98, f"R² = {tr['r_squared']:.3f}\n{p_str}",
                transform=ax.transAxes, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
                fontsize=9)
        ax.set_title(f"{territory_name}: NDVI и тренд", fontsize=12)
        ax.set_ylabel("NDVI mean")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=plot_style["grid_alpha"])
        ax.legend(loc="lower left", fontsize=8)

        ax = axes[1, col]
        colors = [plot_style["anomaly_marker"] if abs(r) > ANOMALY_THRESHOLD
                  else color for r in residuals]
        ax.bar(years, residuals, color=colors, alpha=0.75, width=0.7)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.axhline(y=ANOMALY_THRESHOLD, color="red", linestyle=":", alpha=0.5)
        ax.axhline(y=-ANOMALY_THRESHOLD, color="red", linestyle=":", alpha=0.5)
        ax.set_title(f"{territory_name}: residuals", fontsize=12)
        ax.set_xlabel("Год")
        ax.set_ylabel("Residual")
        ax.grid(True, alpha=plot_style["grid_alpha"])
    fig.suptitle("Тренды NDVI и аномалии (красным = |residual| > 0.1)",
                 fontsize=13, y=1.00)
    _finalize(fig, out_path, verbose)


def _log_verdict(df_f2, analysis, verbose) -> list:
    if df_f2.empty:
        return []
    logger.info("=" * 70)
    logger.info("ВЕРДИКТ F2")
    logger.info("=" * 70)
    issues: list = []

    for territory_name in TERRITORIES:
        subset = df_f2[df_f2["territory"] == territory_name]
        low_cov = subset[subset["coverage_pct"] < 90]
        if not low_cov.empty:
            years = low_cov["year"].astype(int).tolist()
            logger.warning(f"[{territory_name}] coverage <90%: {years}")
            issues.append(f"{territory_name}: низкое покрытие {years}")
        else:
            logger.success(f"[{territory_name}] ✅ coverage ≥90% во все годы")

    for territory_name in TERRITORIES:
        subset = df_f2[df_f2["territory"] == territory_name]
        high_cloud = subset[subset["cloud_residue_pct"] > 1.0]
        if not high_cloud.empty:
            years = high_cloud["year"].astype(int).tolist()
            logger.warning(f"[{territory_name}] cloud_residue >1%: {years}")
            issues.append(f"{territory_name}: остатки облаков {years}")
        else:
            logger.success(f"[{territory_name}] ✅ cloud_residue ≤1% во все годы")

    for territory_name in TERRITORIES:
        subset = df_f2[df_f2["territory"] == territory_name]
        unusual = subset[(subset["NDVI_mean"] < 0.3) | (subset["NDVI_mean"] > 0.8)]
        if not unusual.empty:
            years = unusual["year"].astype(int).tolist()
            logger.warning(f"[{territory_name}] необычный NDVI: {years}")
        else:
            logger.success(f"[{territory_name}] ✅ NDVI_mean в 0.3–0.8")

    logger.info("Валидность обучающих точек (целевые годы):")
    for territory_name in TERRITORIES:
        subset = df_f2[(df_f2["territory"] == territory_name)
                       & (df_f2["is_target"])]
        min_val_pct, min_val_year = 100, None
        for _, row in subset.iterrows():
            val = row.get("training_validation")
            if val and val["valid_pct"] < min_val_pct:
                min_val_pct = val["valid_pct"]
                min_val_year = int(row["year"])
        if min_val_year is None:
            continue
        if min_val_pct >= 70:
            logger.success(f"[{territory_name}] ✅ минимум валидных точек: "
                           f"{min_val_pct:.0f}% ({min_val_year})")
        elif min_val_pct >= 50:
            logger.warning(f"[{territory_name}] ⚠️  минимум валидных точек: "
                           f"{min_val_pct:.0f}% ({min_val_year})")
            issues.append(f"{territory_name}: мало валидных точек в {min_val_year}")
        else:
            logger.error(f"[{territory_name}] ❌ критично мало валидных точек: "
                         f"{min_val_pct:.0f}% ({min_val_year})")
            issues.append(f"{territory_name}: критично мало точек в {min_val_year}")

    if not issues:
        logger.success("✅ F2 ПРОЙДЕН: все композиты и точки качественные")
    else:
        logger.warning(f"⚠️  F2: обнаружено {len(issues)} предупреждений")
        for issue in issues:
            logger.warning(f"  • {issue}")
    return issues
