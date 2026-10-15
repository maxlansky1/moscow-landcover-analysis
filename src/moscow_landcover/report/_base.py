"""Приведение результатов к единому виду (ключи годов/площадей — int)."""
from __future__ import annotations


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


__all__ = ["_normalize_results"]
