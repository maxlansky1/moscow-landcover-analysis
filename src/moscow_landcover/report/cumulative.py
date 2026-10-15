"""Накопительная таблица прогонов (cumulative_stats.csv)."""
from __future__ import annotations

from loguru import logger

from ..config import TERRITORIES
from ..stats import append_to_history, show_cumulative_table


def _update_cumulative(results_by_year, paths, verbose) -> None:
    history_file = paths["INTERMEDIATE"] / "run_history.json"
    cumulative_file = paths["TABLES"] / "cumulative_stats.csv"
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
