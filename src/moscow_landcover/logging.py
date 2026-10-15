"""Настройка логирования (loguru) и мелкие утилиты окружения."""
from __future__ import annotations

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


def setup_logging(logs_dir: Path | str,
                  stdout_level: str = "INFO",
                  file_level: str = "DEBUG",
                  run_id: Optional[str] = None) -> Path:
    """Настраивает loguru: stderr (кратко) + файл (подробно).

    Возвращает путь к созданному лог-файлу.
    """
    global _configured

    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_id or run_timestamp()
    log_file = logs_dir / f"run_{run_id}.log"

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


def configure_matplotlib() -> None:
    """Единые настройки matplotlib (для скриптов; в Colab дополнительно inline)."""
    import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3


def get_logger():
    """Доступ к общему логгеру (loguru — синглтон)."""
    return logger
