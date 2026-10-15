"""Эталонный прогон: заморозка результатов и сравнение с допуском.

Заменяет зашитую в отчёт таблицу `BASELINE_11`. Эталон хранится рядом с
промежуточными результатами (`03_Intermediate/reference.json`) и создаётся
функцией `freeze_reference`. `compare` сверяет новый прогон с эталоном в
пределах допуска — это страховка от незапланированного сдвига чисел при
рефакторинге.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger

REFERENCE_FILENAME = "reference.json"


def default_reference_path(paths: dict) -> Path:
    """Стандартный путь эталона внутри дерева проекта."""
    return Path(paths["INTERMEDIATE"]) / REFERENCE_FILENAME


def freeze_reference(results_by_year: dict, path: Path | str) -> Path:
    """Сохраняет текущие результаты как эталон."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": results_by_year}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.success(f"Эталон зафиксирован: {path}")
    return path


def load_reference(path: Path | str) -> Optional[dict]:
    """Читает эталон; возвращает None, если файла нет.

    Ключи годов и площадей приводятся к int (после JSON они строки).
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    results = payload.get("results", payload)
    return _normalize(results)


def _normalize(results: dict) -> dict:
    """Приводит ключи годов и площадей к int (устойчиво к JSON-строкам)."""
    normalized: dict = {}
    for territory, years in (results or {}).items():
        normalized[territory] = {}
        for year, result in (years or {}).items():
            if not isinstance(result, dict):
                continue
            result = dict(result)
            areas = result.get("areas") or {}
            result["areas"] = {int(k): v for k, v in areas.items()}
            normalized[territory][int(year)] = result
    return normalized


def compare(results_by_year: dict, reference: dict,
            oa_tol: float = 0.02, area_tol_pct: float = 2.0,
            verbose: bool = True) -> dict:
    """Сверяет прогон с эталоном. Возвращает отчёт о расхождениях."""
    results_by_year = _normalize(results_by_year)
    reference = _normalize(reference)
    mismatches: list = []
    checked = 0
    for territory, years in (reference or {}).items():
        current = (results_by_year or {}).get(territory, {})
        for year, ref in years.items():
            year = int(year)
            cur = current.get(year)
            if cur is None:
                mismatches.append({"territory": territory, "year": year,
                                   "field": "missing", "reference": ref,
                                   "current": None})
                continue
            checked += 1
            if abs(cur["oa"] - ref["oa"]) > oa_tol:
                mismatches.append({"territory": territory, "year": year,
                                   "field": "oa", "reference": ref["oa"],
                                   "current": cur["oa"]})
            if abs(cur["kappa"] - ref["kappa"]) > oa_tol:
                mismatches.append({"territory": territory, "year": year,
                                   "field": "kappa", "reference": ref["kappa"],
                                   "current": cur["kappa"]})
            ref_areas = {int(k): v for k, v in (ref.get("areas") or {}).items()}
            cur_areas = {int(k): v for k, v in (cur.get("areas") or {}).items()}
            for cls_id, ref_area in ref_areas.items():
                cur_area = cur_areas.get(cls_id, 0)
                denom = ref_area if ref_area else 1
                if abs(cur_area - ref_area) / denom * 100 > area_tol_pct:
                    mismatches.append({"territory": territory, "year": year,
                                       "field": f"area_{cls_id}",
                                       "reference": ref_area, "current": cur_area})

    status = "ok" if not mismatches else "mismatch"
    if verbose:
        if status == "ok":
            logger.success(f"✅ Эталон совпал: {checked} прогонов в пределах допуска "
                           f"(OA±{oa_tol}, площадь±{area_tol_pct}%)")
        else:
            logger.warning(f"⚠️  Расхождений с эталоном: {len(mismatches)} "
                           f"(проверено {checked} прогонов)")
            for m in mismatches:
                logger.warning(f"  • {m['territory']} {m['year']} {m['field']}: "
                               f"эталон={m['reference']}, сейчас={m['current']}")
    return {"status": status, "checked": checked, "mismatches": mismatches}
