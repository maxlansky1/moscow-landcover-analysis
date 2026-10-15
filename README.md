# Moscow Land Cover Analysis (2000–2026)

Воспроизводимый анализ динамики земного покрова Москвы по снимкам Landsat
(Collection 2 Level 2) в Google Earth Engine. Результат — классифицированные
карты за целевые годы, таблицы площадей, карты разницы, графики и отчёт.

Репозиторий: <https://github.com/maxlansky1/moscow-landcover-analysis>

## Возможности

- Два независимых временных ряда:
  - **Историческая Москва** (границы до 2012): 2000, 2007, 2013, 2019, 2026;
  - **Большая Москва** (с Новой Москвой): 2013, 2019, 2026.
- Медианные летние композиты Landsat 5/7 (2000–2012) и Landsat 8/9 (2013–2026).
- 15 признаков: 6 спектральных каналов + 7 индексов + 2 текстурных (GLCM).
- Random Forest с обучающими метками ESA WorldCover 2021.
- Валидация (OA, Kappa, матрица ошибок, per-class метрики).
- Расчёт площадей, карты разницы, графики, экспорт в Excel.

Полное методологическое описание — в `Снимки_Москвы_спека_4.md`.

## Структура

```
src/moscow_landcover/   Python-пакет (вся логика)
experiments/            изолированные эксперименты (HLS)
notebooks/              тонкие Colab-драйверы
configs/                example.yaml (без секретов), local.yaml (в .gitignore)
tests/                  юнит-тесты
```

Пакет разбит на модули: `config`, `logging`, `gee`, `boundaries`,
`preprocess`, `composite`, `points`, `landsat`, `model`, `pipeline`,
`stats`, `export`, `diagnostics`, `viz`, `cli`.

## Установка

### Локально (разработка)

```bash
python -m pip install -e ".[dev]"
```

### Google Colab

```python
!git clone https://github.com/maxlansky1/moscow-landcover-analysis.git
%cd moscow-landcover-analysis
!pip install -e .
```

Готовый драйвер — `notebooks/colab_runner.ipynb`.

## Настройка

- Скопируйте `configs/example.yaml` в `configs/local.yaml` и при необходимости
  поменяйте параметры и путь `paths.base` (папка проекта на Google Диске).
- Секреты не коммитятся:
  - идентификатор GCP-проекта — переменная окружения `MLC_EE_PROJECT`
    (или `EE_PROJECT`), либо Colab Secret;
  - `configs/local.yaml` и `.env` — в `.gitignore`.

## Запуск

Единая точка входа `mlc`:

```bash
mlc run          --config configs/local.yaml   # полный прогон целевых лет
mlc diagnostics  --config configs/local.yaml   # доступность снимков и диагностика
mlc report       --config configs/local.yaml   # сводка метрик
mlc export       --config configs/local.yaml   # экспорт классификаций
```

Порядок инициализации (в драйвере/скрипте):

```python
from moscow_landcover.config import load_config
from moscow_landcover.logging import setup_logging
from moscow_landcover import gee
from moscow_landcover.pipeline import run

cfg = load_config("configs/local.yaml")
setup_logging(cfg.paths["LOGS"])
gee.authenticate()
gee.initialize()
results = run(cfg, check=False)
```

## Тесты

```bash
pytest
```

Часть пайплайна выполняется на стороне Google Earth Engine и требует
аутентификации; юнит-тесты покрывают детерминированную логику (конфигурацию,
построение путей, форматирование статистик) и импортируемость модулей.

## Лицензия

MIT — см. `LICENSE`.
