"""Тест HLS: Landsat (L30) vs Sentinel-2 (S30) — изолированный эксперимент.

НЕ трогает основной пайплайн: использует собственные функции работы с HLS,
но те же 15 признаков, классы, ANALYSIS_CRS и ESA-точки.
"""
from __future__ import annotations

import importlib.util
import time
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..boundaries import load_boundaries
from ..composite import _add_indices, add_texture_bands
from ..config import AnalysisConfig, Config

TEST_YEARS = [2019, 2020, 2026]
HLS_L30 = "NASA/HLS/HLSL30/v002"
HLS_S30 = "NASA/HLS/HLSS30/v002"

HLS_BAND_MAP = {
    "blue": "B2", "green": "B3", "red": "B4",
    "nir": "B5", "swir1": "B6", "swir2": "B7",
}
HLS_BAND_MAP_S30 = dict(HLS_BAND_MAP, nir="B8A")

HLS_SEASON_START = "06-01"
HLS_SEASON_END = "09-01"


def _have_matplotlib() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


def mask_hls(image):
    """Маска HLS через Fmask (биты 1–4). Вода (бит 5) НЕ маскируется."""
    fmask = image.select("Fmask")
    invalid = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
    clear = fmask.bitwiseAnd(invalid).eq(0)
    return image.updateMask(clear)


def prep_hls_l30(image):
    """HLS Landsat: переименование B2..B7 → blue..swir2."""
    return (
        mask_hls(image)
        .select(
            [HLS_BAND_MAP["blue"], HLS_BAND_MAP["green"], HLS_BAND_MAP["red"],
             HLS_BAND_MAP["nir"], HLS_BAND_MAP["swir1"], HLS_BAND_MAP["swir2"]],
            ["blue", "green", "red", "nir", "swir1", "swir2"],
        )
        .set("hls_sensor", "L30")
    )


def prep_hls_s30(image):
    """HLS Sentinel-2: B2,B3,B4,B8A,B11,B12 → blue..swir2."""
    return (
        mask_hls(image)
        .select(
            [HLS_BAND_MAP_S30["blue"], HLS_BAND_MAP_S30["green"],
             HLS_BAND_MAP_S30["red"], HLS_BAND_MAP_S30["nir"],
             HLS_BAND_MAP_S30["swir1"], HLS_BAND_MAP_S30["swir2"]],
            ["blue", "green", "red", "nir", "swir1", "swir2"],
        )
        .set("hls_sensor", "S30")
    )


def build_hls_composite(collection_id, boundary, year, prep_func,
                        min_images: int = 3):
    """Медианный композит HLS за лето года. Возвращает (composite, n)."""
    import ee

    col = (
        ee.ImageCollection(collection_id)
        .filterBounds(boundary)
        .filterDate(f"{year}-{HLS_SEASON_START}", f"{year}-{HLS_SEASON_END}")
        .map(prep_func)
    )
    n = col.size().getInfo()
    if n < min_images:
        logger.warning(f"  {collection_id.split('/')[-2]} {year}: "
                       f"только {n} снимков (<{min_images})")
    if n == 0:
        return None, 0
    return col.median().clip(boundary), n


def add_hls_features(composite):
    """Добавляет 7 индексов + 2 текстурных признака (те же, что в пайплайне)."""
    composite = composite.addBands(_add_indices(composite))
    contrast, entropy = add_texture_bands(composite)
    return composite.addBands([contrast, entropy])


def run_test(cfg: Config, territory: str = "Greater",
             years: Optional[list] = None, verbose: bool = True,
             plots: bool = True, training_points=None) -> dict:
    """Запускает HLS-сравнение L30 vs S30. Возвращает сводку и результаты."""
    import ee

    from .. import points as points_module

    analysis = cfg.analysis
    test_years = list(years) if years is not None else list(TEST_YEARS)

    logger.info("=" * 70)
    logger.info("ТЕСТ: HLS Landsat vs Sentinel-2 (изолированная ячейка)")
    logger.info("=" * 70)
    logger.info(f"Территория: {territory}")
    logger.info(f"Годы:       {test_years}")
    logger.info(f"Коллекции:  {HLS_L30} / {HLS_S30}")

    boundaries, _ = load_boundaries(cfg.paths)
    boundary = boundaries[territory]
    if training_points is None:
        training_points, _ = points_module.generate_training_data(
            boundary=boundary, year=analysis.training_year, analysis=analysis,
            boundary_name=territory, verbose=verbose,
        )

    test_results: list = []
    for year in test_years:
        logger.info("-" * 70)
        logger.info(f"ГОД {year}")
        logger.info("-" * 70)

        t0 = time.time()
        comp_l30, n_l30 = build_hls_composite(
            HLS_L30, boundary, year, prep_hls_l30, analysis.min_images_for_composite)
        comp_s30, n_s30 = build_hls_composite(
            HLS_S30, boundary, year, prep_hls_s30, analysis.min_images_for_composite)
        if comp_l30 is None or comp_s30 is None:
            logger.error(f"  Композиты не созданы (L30={n_l30}, S30={n_s30})")
            continue

        comp_l30 = add_hls_features(comp_l30)
        comp_s30 = add_hls_features(comp_s30)
        logger.info(f"  Композиты: L30={n_l30} снимков, S30={n_s30} снимков "
                    f"({time.time()-t0:.1f} с)")

        train_samples = comp_s30.select(analysis.features).sampleRegions(
            collection=training_points, properties=["class"],
            scale=30, projection=ee.Projection(analysis.analysis_crs),
            geometries=False, tileScale=4,
        )
        total = train_samples.size().getInfo()
        with_random = train_samples.randomColumn("r", seed=analysis.random_seed)
        train = with_random.filter(ee.Filter.lt("r", analysis.train_fraction))
        test = with_random.filter(ee.Filter.gte("r", analysis.train_fraction))
        train_n = train.size().getInfo()
        test_n = test.size().getInfo()
        logger.info(f"  Образцы: всего={total}, train={train_n}, test={test_n}")
        if train_n == 0 or test_n == 0:
            logger.error(f"  Недостаточно образцов — пропускаю год {year}")
            continue

        clf_s30 = _train_rf(train, analysis)
        test_on_l30 = test.classify(clf_s30, outputName="pred")
        acc_l30 = test_on_l30.errorMatrix("class", "pred")
        oa_l30 = acc_l30.accuracy().getInfo()
        kappa_l30 = acc_l30.kappa().getInfo()

        train_l30 = comp_l30.select(analysis.features).sampleRegions(
            collection=training_points, properties=["class"],
            scale=30, projection=ee.Projection(analysis.analysis_crs),
            geometries=False, tileScale=4,
        ).randomColumn("r", seed=analysis.random_seed)
        train_l30 = train_l30.filter(ee.Filter.lt("r", analysis.train_fraction))
        test_l30 = train_l30.filter(ee.Filter.gte("r", analysis.train_fraction))

        clf_l30 = _train_rf(train_l30, analysis)
        test_l30_classified = test_l30.classify(clf_l30, outputName="pred")
        acc_l30_bench = test_l30_classified.errorMatrix("class", "pred")
        oa_l30_bench = acc_l30_bench.accuracy().getInfo()
        kappa_l30_bench = acc_l30_bench.kappa().getInfo()

        test_on_s30 = test.classify(clf_l30, outputName="pred")
        acc_s30 = test_on_s30.errorMatrix("class", "pred")
        oa_s30 = acc_s30.accuracy().getInfo()
        kappa_s30 = acc_s30.kappa().getInfo()

        sample_pts = training_points.randomColumn("r", seed=analysis.random_seed) \
            .sort("r").limit(3000)
        spec_l30 = comp_l30.select(analysis.features).sampleRegions(
            collection=sample_pts, properties=["class"], scale=30,
            projection=ee.Projection(analysis.analysis_crs),
            geometries=False, tileScale=4,
        ).getInfo().get("features", [])
        spec_s30 = comp_s30.select(analysis.features).sampleRegions(
            collection=sample_pts, properties=["class"], scale=30,
            projection=ee.Projection(analysis.analysis_crs),
            geometries=False, tileScale=4,
        ).getInfo().get("features", [])

        df_l30 = pd.DataFrame([f["properties"] for f in spec_l30])
        df_s30 = pd.DataFrame([f["properties"] for f in spec_s30])
        common = pd.DataFrame()
        for band in analysis.features:
            if band in df_l30.columns and band in df_s30.columns:
                common[f"{band}_L30"] = df_l30[band].values[:len(df_s30)]
                common[f"{band}_S30"] = df_s30[band].values[:len(df_l30)]

        test_results.append({
            "year": year,
            "oa_train_S30_apply_L30": oa_l30,
            "kappa_train_S30_apply_L30": kappa_l30,
            "oa_train_L30_apply_L30": oa_l30_bench,
            "kappa_train_L30_apply_L30": kappa_l30_bench,
            "oa_train_L30_apply_S30": oa_s30,
            "kappa_train_L30_apply_S30": kappa_s30,
            "n_L30": n_l30, "n_S30": n_s30,
            "train_n": train_n, "test_n": test_n,
            "common_spectra": common,
        })

        logger.info(f"  РЕЗУЛЬТАТЫ {year}:")
        logger.info(f"    Обучение S30 → применение L30: OA={oa_l30:.4f}, "
                    f"Kappa={kappa_l30:.4f}")
        logger.info(f"    Обучение L30 → применение L30: OA={oa_l30_bench:.4f}, "
                    f"Kappa={kappa_l30_bench:.4f}")
        logger.info(f"    Обучение L30 → применение S30: OA={oa_s30:.4f}, "
                    f"Kappa={kappa_s30:.4f}")
        logger.info(f"    Δ (S30→L30 vs L30→L30): {oa_l30 - oa_l30_bench:+.4f}")

    df_res = _summary(test_results, analysis, territory, plots, verbose)
    verdict = _log_verdict(test_results, verbose)
    return {"df_res": df_res, "results": test_results, "verdict": verdict}


def _train_rf(train, analysis: AnalysisConfig):
    import ee

    return ee.Classifier.smileRandomForest(
        numberOfTrees=analysis.rf_trees,
        minLeafPopulation=analysis.rf_min_leaf,
        bagFraction=analysis.rf_bag_fraction,
    ).train(features=train, classProperty="class",
            inputProperties=analysis.features)


def _summary(test_results, analysis, territory, plots, verbose):
    logger.info("=" * 70)
    logger.info("СВОДКА ТЕСТА HLS")
    logger.info("=" * 70)
    if not test_results:
        logger.error("Тест не дал результатов.")
        return None

    df_res = pd.DataFrame([
        {k: v for k, v in r.items() if k != "common_spectra"}
        for r in test_results
    ])
    logger.info(f"\n{df_res.to_string(index=False)}")

    if plots and _have_matplotlib():
        _plot_oa(df_res, analysis.accuracy_threshold, territory)
        _plot_delta(df_res)
        _plot_spectra(test_results[-1], analysis)
    elif plots:
        logger.warning("matplotlib недоступен — графики HLS пропущены.")
    return df_res


def _plot_oa(df_res, threshold, territory) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_res))
    w = 0.25
    ax.bar(x - w, df_res["oa_train_S30_apply_L30"], w,
           label="S30→L30 (сценарий заказчика)", color="#d62728")
    ax.bar(x, df_res["oa_train_L30_apply_L30"], w,
           label="L30→L30 (бенчмарк)", color="#1f77b4")
    ax.bar(x + w, df_res["oa_train_L30_apply_S30"], w,
           label="L30→S30 (обратный)", color="#2ca02c")
    ax.axhline(y=threshold, color="green", linestyle="--", alpha=0.6,
               label=f"Цель ({threshold})")
    ax.set_xticks(x)
    ax.set_xticklabels(df_res["year"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Overall Accuracy")
    ax.set_title(f"{territory}: Landsat vs Sentinel-2 через HLS")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    plt.close(fig)


def _plot_delta(df_res) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    delta = df_res["oa_train_S30_apply_L30"] - df_res["oa_train_L30_apply_L30"]
    colors = ["green" if d > 0 else "red" for d in delta]
    ax.bar(df_res["year"].astype(str), delta, color=colors, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhline(y=0.03, color="green", linestyle=":", alpha=0.5,
               label="Порог +3 п.п.")
    ax.axhline(y=-0.03, color="red", linestyle=":", alpha=0.5)
    ax.set_ylabel("Δ OA (S30→L30 − L30→L30)")
    ax.set_title("Прирост от обучения на Sentinel-2")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    plt.close(fig)


def _plot_spectra(last_result, analysis) -> None:
    import matplotlib.pyplot as plt

    common = last_result["common_spectra"]
    if common is None or common.empty:
        return
    bands = [f for f in analysis.spectral_bands + analysis.index_bands
             if f"{f}_L30" in common.columns]
    if not bands:
        return
    fig, axes = plt.subplots(1, len(bands), figsize=(3 * len(bands), 4))
    if len(bands) == 1:
        axes = [axes]
    for ax, band in zip(axes, bands):
        l30_vals = common[f"{band}_L30"].dropna()
        s30_vals = common[f"{band}_S30"].dropna()
        ax.scatter(l30_vals, s30_vals, s=3, alpha=0.3)
        lim = [min(l30_vals.min(), s30_vals.min()),
               max(l30_vals.max(), s30_vals.max())]
        ax.plot(lim, lim, "r--", alpha=0.5, label="1:1")
        ax.set_xlabel(f"{band} L30")
        ax.set_ylabel(f"{band} S30")
        ax.set_title(band)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Спектральные различия L30 vs S30 ({last_result['year']})",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    plt.close(fig)


def _log_verdict(test_results, verbose):
    logger.info("=" * 70)
    logger.info("ВЕРДИКТ ТЕСТА HLS")
    logger.info("=" * 70)
    if not test_results:
        logger.error("Вердикт невозможен — нет результатов.")
        return None

    deltas = [r["oa_train_S30_apply_L30"] - r["oa_train_L30_apply_L30"]
              for r in test_results]
    mean_delta = float(np.mean(deltas))
    if mean_delta > 0.03:
        logger.success(f"✅ Прирост OA от Sentinel-2: {mean_delta:+.4f} "
                       f"(> 3 п.п.) — стоит рассмотреть переход на HLS")
    elif mean_delta > 0:
        logger.info(f"Прирост OA: {mean_delta:+.4f} — незначительный, "
                    f"Landsat остаётся оправданным")
    else:
        logger.warning(f"⚠️  OA упал: {mean_delta:+.4f} — обучение на "
                       f"Sentinel-2 НЕ помогает для этой территории")

    logger.info("ВАЖНО: этот тест НЕ влияет на results_by_year и основной пайплайн.")
    return {"mean_delta": mean_delta, "deltas": [float(d) for d in deltas]}
