"""Конфигурация проекта: параметры анализа, карт, статистик, пути и загрузка.

Источник значений — параметрические ячейки C1–C3 ноутбука-прототипа.
Все параметры собраны в dataclass'ы с дефолтами, совпадающими со спекой v1.0.
Личные значения (project_id, base path) переопределяются через YAML/окружение.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML объявлен в зависимостях
    yaml = None


DEFAULT_PROJECT_FOLDER = "Moscow_LandCover_Analysis"
COLAB_MOUNT_POINT = "/content/drive"

# Территории анализа (порядок фиксирован и используется в отчётах/диагностике).
TERRITORIES = ("Historical", "Greater")


# =====================================================================
# Параметры анализа (бывшие C1)
# =====================================================================
@dataclass
class AnalysisConfig:
    # --- Периоды и годы ---
    historical_start: int = 2000
    historical_end: int = 2026
    greater_start: int = 2013
    greater_end: int = 2026
    historical_years: list = field(default_factory=lambda: [2000, 2007, 2013, 2019, 2026])
    greater_years: list = field(default_factory=lambda: [2013, 2019, 2026])
    diff_pairs: list = field(default_factory=lambda: [
        ("Historical", 2000, 2013),
        ("Historical", 2013, 2026),
        ("Greater", 2013, 2026),
        ("Historical", 2000, 2026),
    ])

    # --- Сезонность ---
    season_start: str = "06-01"
    season_end: str = "09-01"
    min_images_for_composite: int = 3
    fallback_offsets: list = field(default_factory=lambda: [-1, 1, -2, 2])

    # --- Спутники ---
    satellites_early: list = field(default_factory=lambda: [
        "LANDSAT/LT05/C02/T1_L2",
        "LANDSAT/LE07/C02/T1_L2",
    ])
    satellites_late: list = field(default_factory=lambda: [
        "LANDSAT/LC08/C02/T1_L2",
        "LANDSAT/LC09/C02/T1_L2",
    ])

    # --- Классы ---
    classes: list = field(default_factory=lambda: [
        "Dense Vegetation",
        "Sparse Vegetation",
        "Built-Up / Bare Soil",
        "Water",
    ])
    classes_ru: list = field(default_factory=lambda: [
        "Плотная растительность",
        "Редкая растительность",
        "Застройка / открытая почва",
        "Вода",
    ])
    colors: list = field(default_factory=lambda: [
        "#017721",  # 0 Dense Vegetation
        "#5dca0c",  # 1 Sparse Vegetation
        "#ffd700",  # 2 Built-Up / Bare Soil
        "#138eff",  # 3 Water
    ])
    class_short: list = field(default_factory=lambda: [
        "Dense", "Sparse", "Built", "Water",
    ])
    class_weights: Optional[dict] = None  # None → все 1.0

    # --- Признаки (15) ---
    spectral_bands: list = field(default_factory=lambda: [
        "blue", "green", "red", "nir", "swir1", "swir2",
    ])
    index_bands: list = field(default_factory=lambda: [
        "NDVI", "NDWI", "NDBI", "BSI", "SAVI", "NDMI", "MNDWI",
    ])
    texture_bands: list = field(default_factory=lambda: ["NIR_contrast", "NIR_entropy"])

    # --- Random Forest ---
    rf_trees: int = 100
    rf_min_leaf: int = 5
    rf_bag_fraction: float = 0.5

    # --- Обучающие данные ---
    training_source: str = "ESA/WorldCover/v200"
    training_year: int = 2021
    points_per_class: int = 2000
    train_fraction: float = 0.7
    random_seed: int = 42
    training_scale: int = 10

    # ESA WorldCover → наши 4 класса
    esa_remap_from: list = field(default_factory=lambda: [10, 20, 30, 40, 90, 50, 60, 80])
    esa_remap_to: list = field(default_factory=lambda: [0, 1, 1, 1, 1, 2, 2, 3])
    esa_remap_nodata: int = 255

    # --- Постобработка ---
    majority_filter: bool = True
    filter_window: int = 3

    # --- Пороги точности ---
    accuracy_threshold: float = 0.85
    kappa_threshold: float = 0.80

    # --- Проекции ---
    analysis_crs: str = "EPSG:32637"  # UTM 37N
    display_crs: str = "EPSG:3857"    # Web Mercator

    # --- Технические параметры GEE ---
    gee_scale: int = 30
    gee_tile_scale: int = 4
    gee_max_pixels: int = 10**13

    # --- Калибровка Collection 2 Level 2 ---
    sr_scale_factor: float = 0.0000275
    sr_offset: float = -0.2

    @property
    def features(self) -> list:
        """Итоговый список 15 признаков в порядке, ожидаемом классификатором."""
        return list(self.spectral_bands) + list(self.index_bands) + list(self.texture_bands)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def resolved_class_weights(self) -> dict:
        if self.class_weights is not None:
            return dict(self.class_weights)
        return {i: 1.0 for i in range(self.n_classes)}


# =====================================================================
# Параметры карт и визуализации (бывшие C2)
# =====================================================================
@dataclass
class MapConfig:
    territories: dict = field(default_factory=lambda: {
        "Historical": {"center": [55.7520, 37.6180], "zoom": 10, "label": "Историческая Москва"},
        "Greater": {"center": [55.6500, 37.3000], "zoom": 9, "label": "Большая Москва"},
        "Boundaries": {"center": [55.70, 37.45], "zoom": 9, "label": "Проверка границ"},
    })
    basemap: str = "HYBRID"
    width: int = 900
    height: int = 600
    scale_ctrl: bool = True
    default_territory: str = "Historical"

    layer_styles: dict = field(default_factory=lambda: {
        "boundary": {"color": "red", "width": 2, "fillOpacity": 0},
        "diff_loss": {"min": 0, "max": 1, "palette": ["#d73027"]},
        "diff_gain": {"min": 0, "max": 1, "palette": ["#1a9850"]},
        "diff_stable": {"min": 0, "max": 1, "palette": ["#fffacd"]},
        "point": {"color": "red", "pointRadius": 8},
    })
    boundary_layer_styles: dict = field(default_factory=lambda: {
        "Historical": {
            "color": "red", "width": 3, "fillColor": "red",
            "fillOpacity": 0.10, "label": "Историческая Москва (до 2012)",
        },
        "Greater": {
            "color": "blue", "width": 3, "fillColor": "blue",
            "fillOpacity": 0.08, "label": "Большая Москва (с Новой Москвой)",
        },
    })
    legend_config: dict = field(default_factory=lambda: {
        "classification": {
            "title": "Земной покров",
            "keys": [
                "Плотная растительность", "Редкая растительность",
                "Застройка / открытая почва", "Вода",
            ],
            "colors": ["#017721", "#5dca0c", "#ffd700", "#138eff"],
            "position": "bottomright",
        },
        "difference": {
            "title": "Изменения",
            "keys": ["Потеря растительности", "Рост растительности", "Без изменений"],
            "colors": ["#d73027", "#1a9850", "#fffacd"],
            "position": "bottomright",
        },
        "boundaries": {
            "title": "Границы Москвы",
            "keys": ["Историческая Москва", "Большая Москва"],
            "colors": ["red", "blue"],
            "position": "bottomright",
        },
    })
    index_palettes: dict = field(default_factory=lambda: {
        "NDVI": ["#8B4513", "#FFFF00", "#00FF00", "#006400"],
        "NDWI": ["#FFFFFF", "#00BFFF", "#00008B"],
        "NDBI": ["#FFFFFF", "#FFA500", "#8B0000"],
        "BSI": ["#FFFFFF", "#D2B48C", "#8B4513"],
        "SAVI": ["#8B4513", "#FFFF00", "#00FF00", "#006400"],
    })
    map_export: dict = field(default_factory=lambda: {
        "crs": "EPSG:32637",
        "dpi": 150,
        "figsize": (14, 9),
        "figsize_pair": (14, 4.5),
        "figsize_wide": (14, 5),
        "figsize_tall": (15, 9),
        "figsize_grid": (14, 8),
        "bbox_inches": "tight",
        "facecolor": "white",
        "font_family": "DejaVu Sans",
        "title_size": 14,
        "legend_size": 10,
        "figure_dpi": 100,
        "figure_figsize": (12, 6),
        "font_size": 11,
        "true_color_vis": {"min": 0.0, "max": 0.3, "gamma": 1.4},
    })
    plot_style: dict = field(default_factory=lambda: {
        "historical_color": "#1f77b4",
        "greater_color": "#ff7f0e",
        "ok_marker": "green",
        "warn_marker": "orange",
        "error_marker": "red",
        "anomaly_marker": "red",
        "trend_linewidth": 2,
        "grid_alpha": 0.3,
        "marker_size": 8,
        "line_alpha": 0.85,
        "line_width": 1.8,
    })

    @property
    def index_vis(self) -> dict:
        return {name: {"min": -1, "max": 1, "palette": pal}
                for name, pal in self.index_palettes.items()}


# =====================================================================
# Параметры статистик и отчётов (бывшие C3)
# =====================================================================
@dataclass
class StatsConfig:
    show: dict = field(default_factory=lambda: {
        "composite_info": True,
        "training_info": True,
        "accuracy": True,
        "confusion_matrix": True,
        "per_class_metrics": True,
        "areas": True,
        "timing": True,
        "warnings": True,
    })
    format: dict = field(default_factory=lambda: {
        "float_precision": 4,
        "area_precision": 1,
        "percent_precision": 1,
        "separator_width": 72,
        "table_col_width": 10,
    })
    accuracy_marks: dict = field(default_factory=lambda: {
        "oa": [(0.85, "🟢"), (0.75, "🟡"), (0.0, "🔴")],
        "kappa": [(0.80, "🟢"), (0.70, "🟡"), (0.0, "🔴")],
    })
    f1_marks: list = field(default_factory=lambda: [
        (0.80, "✅"), (0.60, "⚠️"), (0.0, "❌"),
    ])
    f1_pair_marks: list = field(default_factory=lambda: [
        (0.80, "✅ отлично"), (0.65, "⚠️ приемлемо"), (0.0, "❌ плохо"),
    ])
    f1_min_ok: float = 0.60
    cumulative_columns: list = field(default_factory=lambda: [
        "territory", "year", "OA", "Kappa",
        "Dense_km2", "Sparse_km2", "BuiltUp_km2", "Water_km2",
        "n_scenes", "elapsed_sec",
    ])


# =====================================================================
# Параметры диагностики F1/F2 (бывшие константы diagnostics.py)
# =====================================================================
@dataclass
class DiagConfig:
    sample_points_per_class: int = 300
    validation_seed: int = 42
    anomaly_threshold: float = 0.1
    trend_min_points: int = 3
    low_coverage_pct: float = 90.0
    coverage_ok_pct: float = 95.0
    cloud_ref_pct: float = 20.0
    high_cloud_pct: float = 1.0
    ndvi_min: float = 0.3
    ndvi_max: float = 0.8
    ndvi_bimod_flat: float = 0.05
    ndvi_bimod_bimodal: float = 0.10
    min_valid_points_pct: float = 70.0
    warn_valid_points_pct: float = 50.0
    separability_good: float = 0.15
    separability_warn: float = 0.05


# =====================================================================
# Пути
# =====================================================================
def build_paths(base: os.PathLike | str, analysis: AnalysisConfig,
                create: bool = False) -> dict[str, Path]:
    """Строит словарь всех путей проекта от корня `base`.

    Ключи: BASE/DOCS/DATA/BOUNDARIES/OUTPUT (плюс PLOTS/TABLES/MAPS)/
    OUT_HIST/OUT_GREATER/OUT_DIFF/INTERMEDIATE (плюс DIAG)/LOGS
    и подпапки годов.
    """
    base = Path(base)
    paths: dict[str, Path] = {
        "BASE": base,
        "DOCS": base / "00_Docs",
        "DATA": base / "01_Data",
        "BOUNDARIES": base / "01_Data" / "Boundaries",
        "OUTPUT": base / "02_Output",
        "PLOTS": base / "02_Output" / "plots",
        "TABLES": base / "02_Output" / "tables",
        "MAPS": base / "02_Output" / "maps",
        "OUT_HIST": base / "02_Output" / "Historical_Moscow",
        "OUT_GREATER": base / "02_Output" / "Greater_Moscow",
        "OUT_DIFF": base / "02_Output" / "Difference",
        "INTERMEDIATE": base / "03_Intermediate",
        "DIAG": base / "03_Intermediate" / "diagnostics",
        "LOGS": base / "04_Logs",
    }
    for year in analysis.historical_years:
        paths[f"OUT_HIST_{year}"] = paths["OUT_HIST"] / str(year)
    for year in analysis.greater_years:
        paths[f"OUT_GREATER_{year}"] = paths["OUT_GREATER"] / str(year)
    for territory, y1, y2 in analysis.diff_pairs:
        paths[f"OUT_DIFF_{territory}_{y1}_{y2}"] = paths["OUT_DIFF"] / f"{territory}_{y1}_{y2}"

    if create:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
    return paths


# =====================================================================
# Итоговый бандл конфигурации
# =====================================================================
@dataclass
class Config:
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    map: MapConfig = field(default_factory=MapConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    diag: DiagConfig = field(default_factory=DiagConfig)
    paths: dict = field(default_factory=dict)
    project_id: str = ""
    base_path: Optional[str] = None
    source_path: Optional[str] = None

    @classmethod
    def standard(cls, base: os.PathLike | str | None = None,
                 project_id: str | None = None) -> "Config":
        analysis = AnalysisConfig()
        base = base or os.environ.get("MLC_PROJECT_PATH") or DEFAULT_PROJECT_FOLDER
        project_id = project_id or os.environ.get("MLC_EE_PROJECT") \
            or os.environ.get("EE_PROJECT") or ""
        return cls(
            analysis=analysis,
            map=MapConfig(),
            stats=StatsConfig(),
            diag=DiagConfig(),
            paths=build_paths(base, analysis),
            project_id=project_id,
            base_path=str(base),
        )


def _overlay(target: Any, data: dict) -> None:
    """Рекурсивно перезаписывает поля dataclass'а значениями из dict."""
    if not isinstance(data, dict):
        return
    valid = {f.name for f in fields(target)}
    for key, value in data.items():
        if key not in valid:
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _overlay(current, value)
        else:
            setattr(target, key, value)


def load_config(path: os.PathLike | str | None = None,
                base: os.PathLike | str | None = None,
                project_id: str | None = None) -> Config:
    """Загружает конфигурацию: дефолты + переопределения из YAML.

    Порядок приоритета: аргумент > YAML > переменная окружения > дефолт.
    """
    cfg = Config.standard(base=base, project_id=project_id)

    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Конфигурация не найдена: {path}")
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML не установлен — не могу прочитать конфиг.")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if "analysis" in data:
            _overlay(cfg.analysis, data["analysis"])
        if "map" in data:
            _overlay(cfg.map, data["map"])
        if "stats" in data:
            _overlay(cfg.stats, data["stats"])
        if "diag" in data:
            _overlay(cfg.diag, data["diag"])

        base_path = (data.get("paths") or {}).get("base") or cfg.base_path
        cfg.base_path = base_path
        cfg.paths = build_paths(base_path, cfg.analysis)

        gee_data = data.get("gee") or {}
        if gee_data.get("project_id"):
            cfg.project_id = gee_data["project_id"]

        cfg.source_path = str(path)

    return cfg


def paths_from_config(cfg: Config, create: bool = False) -> dict[str, Path]:
    """Пересобирает пути (например, если base поменялся после загрузки)."""
    cfg.paths = build_paths(cfg.base_path, cfg.analysis, create=create)
    return cfg.paths
