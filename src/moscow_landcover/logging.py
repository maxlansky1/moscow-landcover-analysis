"""Настройка логирования (loguru) и мелкие утилиты окружения."""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

_STDOUT_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "{message}"
)
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)

# Общий «логический» уровень приложения должен быть сконфигурирован
# ровно один раз; повторные вызовы setup_logging переопределяют хендлеры.
_configured = False


def run_timestamp() -> str:
    """Строковая метка времени для имён файлов (как в прототипе)."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _safe_label(label: str) -> str:
    """Приводит метку запуска к безопасному для имени файла виду."""
    return re.sub(r"[^\w.-]+", "_", label.strip()) or "run"


def setup_logging(logs_dir: Path | str,
                  stdout_level: str = "INFO",
                  file_level: str = "DEBUG",
                  run_id: Optional[str] = None,
                  label: Optional[str] = None) -> Path:
    """Настраивает loguru: stderr (кратко) + файл (подробно).

    Файлы раскладываются по папкам дня:
    `<logs_dir>/<YYYY-MM-DD>/run_<HH-MM-SS>[_<label>].log`.

    Возвращает путь к созданному лог-файлу.
    """
    global _configured

    now = datetime.now()
    day_dir = Path(logs_dir) / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_id or now.strftime("%H-%M-%S")
    if label:
        run_id = f"{run_id}_{_safe_label(label)}"
    log_file = day_dir / f"run_{run_id}.log"

    logger.remove()
    logger.add(sys.stderr, level=stdout_level, format=_STDOUT_FORMAT, colorize=True)
    logger.add(
        log_file,
        level=file_level,
        format=_FILE_FORMAT,
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    _configured = True
    return log_file


def start_run_log(paths: dict, label: Optional[str] = None,
                  stdout_level: str = "INFO",
                  file_level: str = "DEBUG") -> Path:
    """Открывает новый лог-файл на текущий запуск ячейки/шага.

    Тонкая обёртка над `setup_logging` для использования из ноутбука:
    каждая рабочая ячейка начинает с собственного лог-файла.
    """
    return setup_logging(paths["LOGS"], stdout_level=stdout_level,
                         file_level=file_level, label=label)


def configure_matplotlib() -> None:
    """Единые настройки matplotlib (для скриптов; в Colab дополнительно inline)."""
    import matplotlib.pyplot as plt

    from .config import MapConfig

    export = MapConfig().map_export
    plt.rcParams["figure.dpi"] = export["figure_dpi"]
    plt.rcParams["savefig.dpi"] = export["dpi"]
    plt.rcParams["figure.figsize"] = export["figure_figsize"]
    plt.rcParams["font.size"] = export["font_size"]
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3


def get_logger():
    """Доступ к общему логгеру (loguru — синглтон)."""
    return logger
