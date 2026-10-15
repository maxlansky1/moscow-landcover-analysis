"""F1: доступность снимков Landsat 2000–2026 по обеим территориям.

Бывший блок F1 из `diagnostics.py`. Параметры (сезон, минимум снимков,
порог облачности) берутся из `AnalysisConfig`/`DiagConfig`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from .._plotting import finalize_figure, have_matplotlib
from ..boundaries import load_boundaries
from ..config import TERRITORIES, AnalysisConfig, Config
from ..landsat import build_collection
from ..logging import run_timestamp
from ..pipeline import json_safe


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

    Возвращает {territory: DataFrame} и сохраняет diagnostics/f1_availability.json.
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

    if plots and have_matplotlib():
        path = paths["PLOTS"] / "f1_availability.png"
        _plot_availability(f1_data, min_images, path, verbose, cfg)
    elif plots:
        logger.warning("matplotlib недоступен — графики F1 пропущены.")

    f1_json = {
        "run_timestamp": run_timestamp(),
        "years": all_years,
        "min_images_for_composite": min_images,
        "data": {t: json_safe(df.to_dict(orient="records"))
                 for t, df in f1_data.items()},
    }
    f1_file = paths["DIAG"] / "f1_availability.json"
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


def _plot_availability(f1_data, min_images, out_path, verbose, cfg: Config) -> None:
    import matplotlib.pyplot as plt

    export = cfg.map.map_export
    cloud_ref = cfg.diag.cloud_ref_pct
    fig, axes = plt.subplots(2, 1, figsize=export["figsize_tall"], sharex=True)

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
    ax2.axhline(y=cloud_ref, color="red", linestyle="--", alpha=0.7,
                label=f"{cloud_ref:g}% (ориентир)")
    ax2.set_xlabel("Год", fontsize=12)
    ax2.set_ylabel("Медиана облачности (%)", fontsize=12)
    ax2.set_title("Медиана облачности летних снимков", fontsize=13)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(2000, 2027, 2))
    ax2.tick_params(axis="x", rotation=45)

    finalize_figure(fig, out_path, verbose, export["dpi"])
