"""Общие помощники визуализации: наличие matplotlib и сохранение фигур.

Логика вынесена сюда, чтобы `diagnostics` и `report` не дублировали
`_have_matplotlib` / `_finalize`. Параметры экспорта (dpi, figsize)
берутся из `MapConfig.map_export`.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

from loguru import logger


def have_matplotlib() -> bool:
    """True, если matplotlib установлен (tf тяжёлого импорта избегаем)."""
    return importlib.util.find_spec("matplotlib") is not None


def finalize_figure(fig, out_path: Optional[Path] = None, verbose: bool = True,
                    dpi: int = 150) -> None:
    """Делает tight_layout, сохраняет фигуру (если задан путь) и закрывает её."""
    import matplotlib.pyplot as plt

    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        if verbose:
            logger.info(f"График сохранён: {out_path}")
    plt.close(fig)
