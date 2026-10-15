"""Пайплайн: обработка одного года и полный прогон (бывшие E7/H1)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from . import boundaries as boundaries_module
from . import points as points_module
from .composite import create_composite
from .config import AnalysisConfig, Config
from .landsat import build_collection_for_year
from .model import parse_feature_importance, top_features, train_classifier
from .stats import compute_accuracy_metrics


# ---------------------------------------------------------------------
# Обработка одного года
# ---------------------------------------------------------------------
def process_year(boundary, year: int, territory_name: str,
                 analysis: Optional[AnalysisConfig] = None,
                 precomputed_points=None,
                 precomputed_training_info: Optional[dict] = None,
                 points_per_class: Optional[int] = None,
                 verbose: bool = True) -> Optional[dict]:
    """Полный цикл обработки одного года. Возвращает dict результата или None."""
    import ee

    analysis = analysis or AnalysisConfig()
    points_per_class = points_per_class or analysis.points_per_class

    logger.info("=" * 70)
    logger.info(f"[{territory_name}] ГОД {year}: СТАРТ ОБРАБОТКИ")
    logger.info("=" * 70)
    t_start = time.time()

    # ЭТАП 0: коллекция
    collection, sat_ids = build_collection_for_year(year, analysis)
    logger.info(f"[{territory_name}] Год {year}: ЭТАП 0/6 — коллекция: {sat_ids}")

    # ЭТАП 1: композит
    composite, comp_info = create_composite(
        collection=collection, boundary=boundary, year=year,
        territory_name=territory_name, analysis=analysis, verbose=verbose,
    )
    if composite is None:
        logger.error(f"[{territory_name}] Год {year}: композит не создан.")
        return None

    # ЭТАП 2: обучающие точки
    if precomputed_points is not None:
        training_points = precomputed_points
        train_info = precomputed_training_info or {
            "boundary_name": territory_name, "year": year, "source": "precomputed",
        }
    else:
        training_points, train_info = points_module.generate_training_data(
            boundary=boundary, year=year, analysis=analysis,
            boundary_name=territory_name, points_per_class=points_per_class,
            verbose=verbose,
        )
        if training_points is None:
            logger.error(f"[{territory_name}] Год {year}: точки не сгенерированы.")
            return None

    # ЭТАП 3: извлечение спектров + train/test
    training_samples = composite.select(analysis.features).sampleRegions(
        collection=training_points, properties=["class"],
        scale=analysis.gee_scale, projection=ee.Projection(analysis.analysis_crs),
        geometries=False, tileScale=analysis.gee_tile_scale,
    )
    with_random = training_samples.randomColumn("random", seed=analysis.random_seed)
    train = with_random.filter(ee.Filter.lt("random", analysis.train_fraction))
    test = with_random.filter(ee.Filter.gte("random", analysis.train_fraction))
    train_size = train.size().getInfo()
    test_size = test.size().getInfo()
    logger.info(f"[{territory_name}] Год {year}: train={train_size}, test={test_size}.")

    # ЭТАП 4: обучение RF
    classifier = train_classifier(train, analysis)
    feature_importance = parse_feature_importance(classifier, analysis.features,
                                                  territory_name, year)
    if feature_importance and verbose:
        logger.info(f"[{territory_name}] Год {year}: топ-5 признаков:")
        for feat, imp in top_features(feature_importance, 5):
            logger.info(f"    {feat:<16} {imp:.4f}")

    # ЭТАП 5: валидация + классификация
    test_classified = test.classify(classifier)
    accuracy = test_classified.errorMatrix("class", "classification")
    # ОДНО getInfo: OA/Kappa считаются локально из той же матрицы, что и
    # сохраняется, — заголовочные метрики гарантированно согласованы с ней.
    matrix = accuracy.array().getInfo()
    oa, kappa = compute_accuracy_metrics(matrix)
    logger.info(f"[{territory_name}] Год {year}: OA={oa:.4f}, Kappa={kappa:.4f}")
    if oa < analysis.accuracy_threshold:
        logger.warning(f"[{territory_name}] Год {year}: OA ниже порога "
                       f"{analysis.accuracy_threshold}.")
    if kappa < analysis.kappa_threshold:
        logger.warning(f"[{territory_name}] Год {year}: Kappa ниже порога "
                       f"{analysis.kappa_threshold}.")

    classified = composite.select(analysis.features).classify(classifier)
    if analysis.majority_filter:
        classified = classified.focalMode(analysis.filter_window, "square", "pixels")

    # ЭТАП 6: площади
    area_image = ee.Image.pixelArea().divide(1e6).addBands(classified)
    areas_dict = area_image.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
        geometry=boundary.geometry(), scale=analysis.gee_scale,
        crs=analysis.analysis_crs, maxPixels=analysis.gee_max_pixels,
    ).getInfo()

    class_areas, total_area = {}, 0.0
    for item in areas_dict.get("groups", []):
        class_id = int(item["class"])
        class_areas[class_id] = item["sum"]
        total_area += item["sum"]

    for cls_id in sorted(class_areas):
        pct = class_areas[cls_id] / total_area * 100 if total_area > 0 else 0
        logger.info(f"    {analysis.classes[cls_id]:25s}: "
                    f"{class_areas[cls_id]:8.1f} км² ({pct:5.1f}%)")
    logger.info(f"    {'ИТОГО':25s}: {total_area:8.1f} км²")

    elapsed_sec = time.time() - t_start
    logger.success(f"[{territory_name}] Год {year}: ОБРАБОТКА ЗАВЕРШЕНА "
                   f"за {elapsed_sec:.1f} с.")

    return {
        "classified": classified,
        "areas": class_areas,
        "total_area": total_area,
        "oa": oa,
        "kappa": kappa,
        "confusion_matrix": matrix,
        "train_size": train_size,
        "test_size": test_size,
        "territory": territory_name,
        "year": year,
        "sat_ids": sat_ids,
        "n_scenes": comp_info["n_scenes"],
        "used_year": comp_info["used_year"],
        "fallback_used": comp_info["fallback_used"],
        "elapsed_sec": elapsed_sec,
        "composite_info": comp_info,
        "training_info": train_info,
        "feature_importance": feature_importance,
    }


# ---------------------------------------------------------------------
# Сериализация результатов
# ---------------------------------------------------------------------
def json_safe(obj):
    """Преобразует результат в JSON-совместимый вид (убирает ee.Image, np)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if hasattr(v, "getInfo") and not isinstance(v, (int, float, str)):
                out[k] = f"<ee.Image {getattr(v, 'id', 'unknown')}>"
            else:
                out[k] = json_safe(v)
        return out
    if isinstance(obj, list):
        return [json_safe(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _installed_revision() -> Optional[str]:
    """Commit пакета, установленного из git (PEP 610, `direct_url.json`).

    Работает и для обычной установки (`pip install git+...`), и для editable:
    pip записывает точный `commit_id` рядом с метаданными дистрибутива.
    """
    try:
        import importlib.metadata as md

        dist = md.distribution("moscow-landcover")
        direct = dist.read_text("direct_url.json")
        if direct:
            commit = json.loads(direct).get("vcs_info", {}).get("commit_id")
            if commit:
                return commit[:7]
    except Exception:  # noqa: BLE001
        pass
    return None


def _repo_revision() -> Optional[str]:
    """Короткий git-хэш локального checkout (None, если git недоступен)."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=repo_root)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def _git_revision() -> Optional[str]:
    """Ревизия кода: установленный из git commit, иначе хэш локального репо."""
    return _installed_revision() or _repo_revision()


def run_meta() -> dict:
    """Метаданные прогона: версия пакета, Python, git-хэш."""
    from . import __version__

    meta = {"package_version": __version__, "python": sys.version.split()[0]}
    revision = _git_revision()
    if revision:
        meta["git_revision"] = revision
    return meta


def save_results(results: dict, failed: list, path: Path | str,
                 run_timestamp: str = "", meta: Optional[dict] = None) -> Path:
    """Сохраняет промежуточные/итоговые результаты в JSON.

    Если `run_timestamp` не задан — подставляется текущее время; в файл
    добавляется блок `meta` (версия, Python, git-хэш).
    """
    from .logging import run_timestamp as _now

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_timestamp": run_timestamp or _now(),
        "meta": meta if meta is not None else run_meta(),
        "results": json_safe(results),
        "failed": failed,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------
# Полный прогон
# ---------------------------------------------------------------------
def build_run_plan(analysis: AnalysisConfig):
    """План прогона: список (territory, year) по обоим рядам."""
    plan = [("Historical", y) for y in analysis.historical_years]
    plan += [("Greater", y) for y in analysis.greater_years]
    return plan


def run(cfg: Config, plan: Optional[list] = None,
        precomputed_points: Optional[dict] = None,
        territories_map: Optional[dict] = None,
        run_timestamp: str = "",
        max_retries: int = 3, retry_delays=(30, 60, 90),
        save: bool = True, verbose: bool = True):
    """Полный прогон целевых лет обоих рядов.

    Возвращает (results_by_year, failed_runs).
    """
    analysis = cfg.analysis
    paths = cfg.paths
    plan = plan or build_run_plan(analysis)

    # Границы
    if territories_map is None:
        boundaries, areas = boundaries_module.load_boundaries(paths)
    else:
        boundaries = dict(territories_map)
        areas = {k: v.geometry().area(maxError=1).getInfo() / 1e6
                 for k, v in boundaries.items()}

    # Обучающие точки на территорию (кэшируются между годами)
    points_cache = dict(precomputed_points or {})
    info_cache: dict = {}
    for name in {t for t, _ in plan}:
        if name not in points_cache:
            pts, info = points_module.generate_training_data(
                boundary=boundaries[name], year=analysis.training_year,
                analysis=analysis, boundary_name=name, verbose=verbose,
            )
            if pts is None:
                raise RuntimeError(f"Не удалось сгенерировать точки для {name}.")
            points_cache[name] = pts
            info_cache[name] = info

    partial_file = paths["INTERMEDIATE"] / "partial_results.json"
    final_file = paths["INTERMEDIATE"] / "results_by_year.json"

    results_by_year = {"Historical": {}, "Greater": {}}
    failed_runs: list = []

    logger.info(f"План прогона ({len(plan)} итераций): "
                + ", ".join(f"{t} {y}" for t, y in plan))

    for territory_name, year in plan:
        boundary = boundaries[territory_name]
        result, last_error = None, None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"[run] {territory_name} {year}: попытка "
                            f"{attempt}/{max_retries}")
                result = process_year(
                    boundary=boundary, year=year, territory_name=territory_name,
                    analysis=analysis,
                    precomputed_points=points_cache.get(territory_name),
                    precomputed_training_info=info_cache.get(territory_name),
                    verbose=verbose,
                )
                if result is not None:
                    break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logger.error(f"[run] {territory_name} {year}: попытка {attempt} "
                             f"провалилась: {exc}")
            if attempt < max_retries:
                time.sleep(retry_delays[attempt - 1])

        if result is None:
            logger.error(f"[run] {territory_name} {year}: все попытки провалились")
            failed_runs.append({"territory": territory_name, "year": year,
                                "error": last_error})
            if save:
                save_results(results_by_year, failed_runs, partial_file, run_timestamp)
            continue

        area_sum = sum(result["areas"].values())
        boundary_area = areas[territory_name]
        diff_pct = abs(area_sum - boundary_area) / boundary_area * 100
        if diff_pct > 2.0:
            logger.warning(f"[run] {territory_name} {year}: сумма площадей "
                           f"{area_sum:.1f} ≠ границы {boundary_area:.1f} (Δ {diff_pct:.2f}%)")

        results_by_year.setdefault(territory_name, {})[year] = result
        logger.success(f"[run] {territory_name} {year}: OK (OA={result['oa']:.4f}, "
                       f"Kappa={result['kappa']:.4f}, площадь={result['total_area']:.1f} км²)")
        if save:
            save_results(results_by_year, failed_runs, partial_file, run_timestamp)

    if save:
        save_results(results_by_year, failed_runs, final_file, run_timestamp)
        logger.success(f"Результаты сохранены: {final_file.name}")

    if failed_runs:
        logger.warning(f"Провалившихся прогонов: {len(failed_runs)}")
    return results_by_year, failed_runs
