"""Графики H2: столбцы OA, тренд OA и матрицы ошибок (matplotlib)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .._plotting import finalize_figure
from ..config import TERRITORIES, AnalysisConfig, MapConfig


def plot_oa_bars(results_by_year, analysis: AnalysisConfig, map_cfg: MapConfig,
                 out_path, verbose) -> None:
    import matplotlib.pyplot as plt

    plot_style = map_cfg.plot_style
    fig, axes = plt.subplots(1, 2, figsize=map_cfg.map_export["figsize_wide"])
    for ax, territory_name in zip(axes, TERRITORIES):
        subset = results_by_year.get(territory_name, {})
        if not subset:
            ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
            ax.set_title(territory_name)
            continue
        years = sorted(subset)
        oa_vals = [subset[y]["oa"] for y in years]
        color = (plot_style["historical_color"] if territory_name == "Historical"
                 else plot_style["greater_color"])
        bars = ax.bar(years, oa_vals, color=color, alpha=0.85, edgecolor="black")
        ax.axhline(y=analysis.accuracy_threshold, color="green", linestyle="--",
                   alpha=0.6, linewidth=1.5,
                   label=f"Цель ({analysis.accuracy_threshold})")
        for bar, val in zip(bars, oa_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{territory_name}: OA по годам", fontsize=12)
        ax.set_xlabel("Год")
        ax.set_ylabel("OA")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=plot_style["grid_alpha"], axis="y")
    fig.suptitle("Overall Accuracy по целевым годам", fontsize=13, y=1.02)
    finalize_figure(fig, out_path, verbose, map_cfg.map_export["dpi"])


def plot_oa_trend(results_by_year, analysis: AnalysisConfig, map_cfg: MapConfig,
                  out_path, verbose) -> None:
    import matplotlib.pyplot as plt

    plot_style = map_cfg.plot_style
    fig, axes = plt.subplots(1, 2, figsize=map_cfg.map_export["figsize_pair"])
    for ax, territory_name in zip(axes, TERRITORIES):
        subset = results_by_year.get(territory_name, {})
        if not subset:
            continue
        years = sorted(subset)
        oa_vals = [subset[y]["oa"] for y in years]
        color = (plot_style["historical_color"] if territory_name == "Historical"
                 else plot_style["greater_color"])
        ax.plot(years, oa_vals, marker="o", linewidth=2, markersize=8,
                color=color, label="OA")
        if len(years) >= 3:
            coef = np.polyfit(years, oa_vals, 1)
            trend = np.polyval(coef, years)
            ax.plot(years, trend, "--", linewidth=1.5, color="black", alpha=0.6,
                    label=f"Тренд: {coef[0]:+.5f}/год")
            ss_res = np.sum((np.array(oa_vals) - trend) ** 2)
            ss_tot = np.sum((np.array(oa_vals) - np.mean(oa_vals)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            ax.text(0.02, 0.98, f"R² = {r2:.3f}", transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
                    fontsize=10)
        ax.axhline(y=analysis.accuracy_threshold, color="green", linestyle=":",
                   alpha=0.6, label=f"Цель ({analysis.accuracy_threshold})")
        ax.set_title(f"{territory_name}: тренд OA по годам", fontsize=12)
        ax.set_xlabel("Год")
        ax.set_ylabel("OA")
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=plot_style["grid_alpha"])
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Тренд Overall Accuracy (целевые годы)", fontsize=13, y=1.02)
    finalize_figure(fig, out_path, verbose, map_cfg.map_export["dpi"])


def plot_confusion_heatmaps(results_by_year, analysis: AnalysisConfig,
                            map_cfg: MapConfig, out_dir, verbose) -> None:
    import matplotlib.pyplot as plt

    for territory_name in TERRITORIES:
        subset = results_by_year.get(territory_name, {})
        if not subset:
            continue
        years = sorted(subset)
        n_years = len(years)
        if n_years <= 3:
            fig, axes = plt.subplots(1, n_years, figsize=(5 * n_years, 5))
            if n_years == 1:
                axes = [axes]
        else:
            fig, axes = plt.subplots(1, n_years, figsize=(4.5 * n_years, 5))
        for ax, year in zip(axes, years):
            _plot_confusion_heatmap(
                subset[year]["confusion_matrix"], analysis.classes,
                analysis.class_short,
                f"{year}\nOA = {subset[year]['oa']:.3f}", ax,
            )
        fig.suptitle(f"{territory_name}: Confusion Matrix по годам "
                     f"(нормализовано по строкам)", fontsize=13, y=1.02)
        finalize_figure(fig,
                        Path(out_dir) / f"confusion_matrix_{territory_name.lower()}.png",
                        verbose, map_cfg.map_export["dpi"])


def _plot_confusion_heatmap(matrix, classes, short_classes, title, ax) -> None:
    matrix = np.array(matrix)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = matrix / np.where(row_sums > 0, row_sums, 1)
    ax.imshow(matrix_norm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(short_classes, rotation=0, fontsize=9)
    ax.set_yticklabels(short_classes, fontsize=9)
    ax.set_xlabel("Предсказание", fontsize=9)
    ax.set_ylabel("Истина", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for i in range(len(classes)):
        for j in range(len(classes)):
            count = matrix[i, j]
            frac = matrix_norm[i, j]
            ax.text(j, i, f"{count:.0f}\n{frac:.2f}", ha="center", va="center",
                    color="white" if frac > 0.55 else "black", fontsize=8)
