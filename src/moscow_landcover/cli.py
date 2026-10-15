"""Единая точка входа `mlc` (argparse): run | diagnostics | report | export."""
from __future__ import annotations

import argparse
import sys

from .config import load_config, paths_from_config


def _prepare(cfg, create: bool = True):
    """Создаёт дерево путей и настраивает логирование. Возвращает пути."""
    from .logging import setup_logging

    paths = paths_from_config(cfg, create=create)
    setup_logging(paths["LOGS"])
    return paths


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", "-c", default=None,
                        help="Путь к YAML-конфигурации (напр. configs/local.yaml)")
    parser.add_argument("--base", default=None,
                        help="Корень дерева проекта (переопределяет paths.base)")


def cmd_run(args) -> int:
    cfg = load_config(args.config, base=args.base)
    _prepare(cfg)

    from . import gee
    from .pipeline import run

    gee.initialize(cfg.project_id or None)
    if not args.skip_check:
        gee.check()
    results, failed = run(cfg, save=not args.no_save)
    n_ok = sum(len(v) for v in results.values())
    print(f"Готово: {n_ok} прогонов, провалов: {len(failed)}")
    return 1 if failed else 0


def cmd_diagnostics(args) -> int:
    cfg = load_config(args.config, base=args.base)
    _prepare(cfg)

    from . import gee
    from .diagnostics import availability, run_diagnostics

    gee.initialize(cfg.project_id or None)
    f1 = availability(cfg)
    run_diagnostics(cfg, f1)
    print("Диагностика завершена.")
    return 0


def cmd_report(args) -> int:
    cfg = load_config(args.config, base=args.base)
    paths = _prepare(cfg)

    import json
    from .report import summary

    results = None
    final = paths["INTERMEDIATE"] / "results_by_year.json"
    if final.exists():
        with open(final, "r", encoding="utf-8") as f:
            payload = json.load(f)
        results = payload.get("results", payload)
    else:
        print(f"Нет {final} — сначала выполните `mlc run`.", file=sys.stderr)
        return 1
    summary(cfg, results)
    print("Отчёт собран.")
    return 0


def cmd_export(args) -> int:
    cfg = load_config(args.config, base=args.base)
    paths = _prepare(cfg)

    import json
    from . import boundaries as boundaries_module
    from . import export as export_module

    final = paths["INTERMEDIATE"] / "results_by_year.json"
    if not final.exists():
        print(f"Нет {final} — сначала выполните `mlc run`.", file=sys.stderr)
        return 1
    with open(final, "r", encoding="utf-8") as f:
        results = json.load(f).get("results", {})
    boundaries, _ = boundaries_module.load_boundaries(paths)

    if args.mode == "drive":
        export_module.export_classifications_to_drive(
            results, boundaries, drive_folder=args.folder)
    else:
        export_module.export_classifications_to_assets(
            results, boundaries, gee_user=args.gee_user, gee_folder=args.folder)
    print("Экспорт запущен.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlc",
        description="Moscow Land Cover Analysis — пайплайн и отчёты.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Полный прогон целевых лет")
    _common(p_run)
    p_run.add_argument("--skip-check", action="store_true",
                       help="Не выполнять gee.check перед прогоном")
    p_run.add_argument("--no-save", action="store_true",
                       help="Не сохранять промежуточные JSON")
    p_run.set_defaults(func=cmd_run)

    p_diag = sub.add_parser("diagnostics", help="Доступность снимков и диагностика")
    _common(p_diag)
    p_diag.set_defaults(func=cmd_diagnostics)

    p_rep = sub.add_parser("report", help="Сводка метрик")
    _common(p_rep)
    p_rep.set_defaults(func=cmd_report)

    p_exp = sub.add_parser("export", help="Экспорт классификаций")
    _common(p_exp)
    p_exp.add_argument("--mode", choices=["drive", "assets"], default="drive")
    p_exp.add_argument("--folder", default="Moscow_LandCover_Classifications")
    p_exp.add_argument("--gee-user", default="your_username")
    p_exp.set_defaults(func=cmd_export)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
