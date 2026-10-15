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
src/moscow_landcover/experiments/   изолированные эксперименты (HLS)
notebooks/              тонкие Colab-драйверы
configs/                example.yaml (без секретов), local.yaml (в .gitignore)
tests/                  юнит-тесты
```

Пакет разбит на модули: `config`, `logging`, `gee`, `boundaries`,
`preprocess`, `composite`, `points`, `landsat`, `model`, `pipeline`,
`stats`, `export`, `reference`, `viz`, а также пакеты `diagnostics/`
(F1 `availability`, F2 `spectral`) и `report/` (таблицы, графики, вердикт,
накопительная таблица).

### Артефакты (папка проекта на Диске)

Все результаты лежат **только** на Google Диске (в `.gitignore`), разложены
по типу:

```
02_Output/
  plots/   графики (PNG): oa_by_year, oa_trend, confusion_matrix_*,
           f1_availability, f2_ndvi_timeseries, f2_ndvi_distribution, f2_ndvi_trends
  tables/  таблицы (CSV): cumulative_stats.csv
  maps/    статические карты (Блок I, пока пусто)
  Historical_Moscow/<год>/, Greater_Moscow/<год>/, Difference/<территория_г1_г2>/
03_Intermediate/
  results_by_year.json, partial_results.json, run_history.json, reference.json
  diagnostics/  f1_availability.json, f2_composite.json
04_Logs/<YYYY-MM-DD>/run_HH-MM-SS[_<метка>].log
```

## Установка

### Локально (разработка)

```bash
python -m pip install -e ".[dev]"
# один раз в репозитории — чистить выводы ноутбуков при коммите:
nbstripout --install
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

Вся работа — из ноутбука `notebooks/colab_runner.ipynb` (в Colab) или через
Python API пакета. Порядок инициализации:

```python
from moscow_landcover.config import load_config, paths_from_config
from moscow_landcover.logging import start_run_log
from moscow_landcover import gee
from moscow_landcover.pipeline import run
from moscow_landcover.report import summary

cfg = load_config("configs/local.yaml")
paths = paths_from_config(cfg, create=True)
start_run_log(paths, label="setup")   # новый лог-файл на запуск шага

gee.authenticate()
gee.initialize()
results, failed = run(cfg)                 # все целевые годы
# быстрый smoke-тест: run(cfg, plan=[("Historical", 2026)])
summary(cfg, results)                      # сводка H2
```

### Эталон (воспроизводимость)

Зашитая таблица baseline убрана; вместо неё — замороженный эталон:

```python
from moscow_landcover import reference

# один раз, когда прогон устраивает:
reference.freeze_reference(results, reference.default_reference_path(paths))
# при последующих прогонах:
reference.compare(results, reference.load_reference(reference.default_reference_path(paths)))
```

Эталон (`03_Intermediate/reference.json`) лежит рядом с результатами и
содержит `OA`, `Kappa` и площади по каждому году; `compare` сверяет в пределах
допуска. Накопительная таблица — `02_Output/tables/cumulative_stats.csv`.

## Тесты

```bash
pytest
```

Часть пайплайна выполняется на стороне Google Earth Engine и требует
аутентификации; юнит-тесты покрывают детерминированную логику (конфигурацию,
построение путей, форматирование статистик) и импортируемость модулей.

## Лицензия

MIT — см. `LICENSE`.
