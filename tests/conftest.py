"""Общие фикстуры и стабы внешних зависимостей (ee, geemap) для тестов.

`ee` и `geemap` не устанавливаются в CI/локально, но нужны лишь во время
выполнения GEE-кода, а не при импорте модулей. Стабы позволяют импортировать
пакет и проверять сигнатуры/чистую логику без доступа к Earth Engine.
"""
from __future__ import annotations

import importlib.util
import sys
import types

import pytest


class _Any:
    """Универсальный заглушечный объект для fluent-API GEE."""

    def __getattr__(self, name):  # noqa: D105
        return _Any()

    def __call__(self, *args, **kwargs):  # noqa: D102
        return _Any()

    def __getitem__(self, key):  # noqa: D105
        return _Any()

    def __iter__(self):  # noqa: D105
        return iter(())

    def __bool__(self):  # noqa: D105
        return True


def _install_stub(name: str) -> None:
    if importlib.util.find_spec(name) is not None:
        return  # реальный пакет есть — не трогаем
    module = types.ModuleType(name)
    module.__getattr__ = lambda attr: _Any()  # PEP 562
    sys.modules[name] = module


for _name in ("ee", "geemap"):
    _install_stub(_name)


@pytest.fixture
def tmp_base(tmp_path):
    """Временный корень дерева проекта."""
    return tmp_path / "Moscow_LandCover_Analysis"
