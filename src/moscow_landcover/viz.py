"""Визуализация: интерактивные карты geemap и статические графики (B3/C2/D2)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from .config import AnalysisConfig, MapConfig


def create_base_map(territory: str = "Historical",
                    map_cfg: Optional[MapConfig] = None):
    """Создаёт базовую интерактивную карту geemap для территории."""
    import geemap

    map_cfg = map_cfg or MapConfig()
    if territory not in map_cfg.territories:
        raise ValueError(f"Неизвестная территория: {territory}. "
                         f"Допустимо: {list(map_cfg.territories)}")
    c = map_cfg.territories[territory]
    m = geemap.Map(center=c["center"], zoom=c["zoom"],
                   width=map_cfg.width, height=map_cfg.height,
                   scale_ctrl=map_cfg.scale_ctrl)
    m.add_basemap(map_cfg.basemap)
    return m


def add_boundary_layer(m, boundary, label: str = "Граница", style: Optional[dict] = None):
    """Добавляет слой границы территории."""
    style = style or {"color": "red", "width": 2, "fillOpacity": 0}
    m.addLayer(boundary, style, label)
    return m


def add_classification_layer(m, classified, year, label: Optional[str] = None,
                             analysis: Optional[AnalysisConfig] = None,
                             map_cfg: Optional[MapConfig] = None):
    """Добавляет слой классификации земного покрова."""
    analysis = analysis or AnalysisConfig()
    map_cfg = map_cfg or MapConfig()
    label = label or f"Классификация {year}"
    m.addLayer(classified, {"min": 0, "max": analysis.n_classes - 1,
                            "palette": analysis.colors}, label)
    return m


def add_index_layer(m, composite, index_name: str,
                    map_cfg: Optional[MapConfig] = None):
    """Добавляет слой индекса (NDVI, NDWI, NDBI, BSI, SAVI, NDMI, MNDWI)."""
    map_cfg = map_cfg or MapConfig()
    vis = map_cfg.index_vis
    if index_name not in vis:
        raise ValueError(f"Индекс {index_name} не определён в INDEX_VIS.")
    m.addLayer(composite.select(index_name), vis[index_name], index_name)
    return m


def add_legend(m, legend_type: str = "classification",
               map_cfg: Optional[MapConfig] = None):
    """Добавляет легенду через встроенный geemap.Map.add_legend()."""
    map_cfg = map_cfg or MapConfig()
    if legend_type not in map_cfg.legend_config:
        raise ValueError(f"Легенда '{legend_type}' не определена. "
                         f"Допустимо: {list(map_cfg.legend_config)}")
    cfg = map_cfg.legend_config[legend_type]
    m.add_legend(title=cfg["title"], keys=cfg["keys"],
                 colors=cfg["colors"], position=cfg["position"])
    return m


def true_color_rgb(image):
    """Готовит true-color RGB из калиброванного изображения (red, green, blue)."""
    return image.select(["red", "green", "blue"])


def save_figure(fig, output_path: Path | str,
                map_cfg: Optional[MapConfig] = None) -> Path:
    """Сохраняет matplotlib-фигуру в PNG с настройками экспорта проекта."""
    map_cfg = map_cfg or MapConfig()
    export = map_cfg.map_export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=export["dpi"], bbox_inches=export["bbox_inches"],
                facecolor=export["facecolor"])
    logger.info(f"Рисунок сохранён: {output_path}")
    return output_path
