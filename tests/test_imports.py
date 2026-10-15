"""Импортный smoke-тест: пакет и все модули импортируются без ee/geemap."""
from __future__ import annotations

import importlib

import pytest

MODULES = [
    "moscow_landcover",
    "moscow_landcover.config",
    "moscow_landcover.logging",
    "moscow_landcover.gee",
    "moscow_landcover.boundaries",
    "moscow_landcover.preprocess",
    "moscow_landcover.composite",
    "moscow_landcover.points",
    "moscow_landcover.landsat",
    "moscow_landcover.model",
    "moscow_landcover.stats",
    "moscow_landcover.export",
    "moscow_landcover.pipeline",
    "moscow_landcover.reference",
    "moscow_landcover.diagnostics",
    "moscow_landcover.report",
    "moscow_landcover.viz",
    "moscow_landcover.experiments.hls",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    assert importlib.import_module(name) is not None


def test_public_api_exports():
    import moscow_landcover as m

    for name in m.__all__:
        assert hasattr(m, name)


def test_run_meta_has_git_revision():
    from moscow_landcover.pipeline import run_meta

    meta = run_meta()
    assert meta["package_version"]
    assert "python" in meta
    # в репозитории git доступен → хэш должен подставляться
    assert meta.get("git_revision")


def test_git_revision_installed_first_then_repo():
    from moscow_landcover.pipeline import (
        _git_revision,
        _installed_revision,
        _repo_revision,
    )

    installed = _installed_revision()
    # локально пакет не установлен из git → None; если установлен — короткий sha
    assert installed is None or all(c in "0123456789abcdef" for c in installed)
    assert _git_revision()  # в репозитории хотя бы repo-хэш
    assert _repo_revision() or _installed_revision()
