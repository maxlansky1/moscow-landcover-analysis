# План рефакторинга: `Moscow_LandCover_Analysis.ipynb` → Python-пакет

Репозиторий: https://github.com/maxlansky1/moscow-landcover-analysis

## 1. Назначение документа

Это пошаговый план работ по превращению единственного ноутбука
`Moscow_LandCover_Analysis.ipynb` в полноценный, поддерживаемый Python-проект
(`moscow_landcover`). Цель — сохранить текущее поведение и численные
результаты, но получить чистую структуру, которую удобно править через
OpenCode на сервере и запускать в Google Colab.

## 2. Контекст и мотивация

- Ноутбук разросся (~33 ячейки, ~6 500 строк кода, блоки A–H). В нём тяжело
  ориентироваться, диффы в git нечитаемы, а логика перемешана с Colab-спецификой.
- Все вычисления выполняет Google Earth Engine (GEE) на стороне Google; Colab —
  это лишь среда запуска. Данные — публичные спутниковые снимки.
- Заказчику (другу) нужны **код и результаты**; владение остаётся у автора.
- Целевой рабочий цикл: правки в `.py` на сервере через OpenCode → `git push` →
  Colab делает `git pull` и запускает. Google Диск остаётся хранилищем данных.

### Принятые решения

- Источник истины — **git** (публичный репозиторий, лицензия **MIT**).
- Подход — **полный Python-пакет** + тонкий Colab-драйвер + ноутбук для интерактива.
- Секреты (id GEE-проекта, пути) — **не в коде** (Colab Secrets / переменные окружения).
- `nbstripout` — для коммита ноутбуков без выводов ячеек.

## 3. Текущее состояние

Ноутбук `Moscow_LandCover_Analysis.ipynb`, kernel `python3`, 26 code-ячеек и
7 markdown-ячеек, организованных в блоки:

| Блок | Ячейки | Содержание |
|---|---|---|
| A. Окружение | A1–A4 | установка пакетов, монтирование Drive, конфигурация путей, импорты + логирование (loguru) |
| B. Earth Engine | B1–B3 | аутентификация, `ee.Initialize` с проектом, проверка + визуализация |
| C. Параметры анализа | C1–C3 | параметры анализа, параметры карт, параметры статистик |
| D. Границы | D1–D2 | загрузка GeoJSON (историческая/Большая Москва), визуализация |
| E. Функции обработки | E1–E7 | маски/гармонизация/калибровка Landsat, композит, обучающие точки (ESA WorldCover), `process_year`, выбор коллекции по году |
| F. Диагностика | F1, F2 (+HLS) | доступность снимков 2000–2026, единый проход по годам × территориям, тест HLS |
| H. Полный запуск | H1–H3 | полный прогон целевых лет, сводка метрик, экспорт классификаций (заглушка) |

Примечание: блок G отсутствует — нумерация переходит с F на H.

### Используемые зависимости (из A1)

Пиннинг в текущем ноутбуке: `loguru==0.7.3`, `earthengine-api==1.7.39`,
`geemap==0.38.8`, `openpyxl==3.1.5`, `pandas==2.2.3`, `matplotlib==3.10.0`.
Также используются (без явного пина): `numpy`, `scipy`, `tqdm`.

## 4. Целевая архитектура

```
moscow-landcover-analysis/
  README.md                 # назначение, установка, запуск, бейдж "Open in Colab"
  LICENSE                   # MIT
  REFACTORING.md            # этот документ
  pyproject.toml            # метаданные, зависимости, CLI `mlc`
  requirements.txt          # пиннинг версий (для Colab)
  .gitignore                # .ipynb_checkpoints/, данные, .env
  configs/
    example.yaml            # параметры без секретов
  src/moscow_landcover/
    __init__.py
    logging.py              # настройка loguru (бывший A4)
    config.py               # загрузка/валидация конфига, пути (A3, C1–C3)
    gee.py                  # auth/init/check GEE (B1–B3)
    boundaries.py           # загрузка границ (D1)
    preprocess.py           # маски/гармонизация/калибровка (E1)
    composite.py            # медианный композит (E3)
    points.py               # обучающие точки ESA WorldCover (E5)
    landsat.py              # build_collection_for_year (E7)
    model.py                # обучение классификатора, parse_feature_importance (E7, H1)
    pipeline.py             # process_year, run (H1)
    stats.py                # форматтеры, метрики, история (C3)
    report.py               # итоговая сводка H2 + графики
    diagnostics.py          # доступность/диагностика (F1, F2)
    export.py               # экспорт классификаций (H3)
    viz.py                  # карты/графики (B3, D2, C2)
    cli.py                  # argparse: `mlc run|diagnostics|report|export`
    experiments/
      __init__.py
      hls.py                # тест HLS L30 vs S30 (бывшая ячейка F/HLS)
  notebooks/
    colab_runner.ipynb      # тонкий драйвер (A2, B1–B2, запуск)
    exploration.ipynb       # интерактив: карты/графики/проверки (запланирован)
  tests/
    conftest.py             # стабы ee/geemap, общие фикстуры
    test_config.py
    test_preprocess.py
    test_stats.py
    test_diagnostics.py
    test_imports.py
```

## 5. Маппинг ячеек ноутбука → модули

| Ячейка | Куда переносится |
|---|---|
| A1 установка пакетов | `pyproject.toml` / `requirements.txt` |
| A2 монтирование Drive | `notebooks/colab_runner.ipynb` (Colab-специфика) |
| A3 пути проекта | `config.py` (функция построения путей) |
| A4 импорты + логирование | `logging.py`, `__init__.py` |
| B1 auth GEE | `gee.authenticate()` (вызов из драйвера) |
| B2 init GEE | `gee.initialize(project)` |
| B3 проверка + визуализация | `gee.check()` + `viz.py` |
| C1 параметры анализа | `config.py` (`AnalysisConfig`) |
| C2 параметры карт | `config.py` (`MapConfig`) |
| C3 параметры статистик | `config.py` (`StatsConfig`) + `stats.py` |
| D1 загрузка границ | `boundaries.load()` |
| D2 визуализация границ | `viz.py` + `boundaries.py` |
| E1 маски/гармонизация/калибровка | `preprocess.py` |
| E2 тест предобработки | `tests/` / `notebooks/exploration.ipynb` |
| E3 медианный композит | `composite.create_composite()` |
| E4 тест композита | `tests/` / `notebooks/exploration.ipynb` |
| E5 обучающие точки ESA | `points.generate_training_points()` |
| E6 тест точек | `tests/` / `notebooks/exploration.ipynb` |
| E7 collection + feature importance | `landsat.py` + `model.py` |
| F1 доступность снимков | `diagnostics.availability()` |
| F2 диагностика композитов и точек | `diagnostics.run()` |
| F/HLS тест | `experiments/hls.py` |
| H1 полный прогон | `pipeline.run()` |
| H2 сводка метрик | `report.summary()` (детерминированные части — `stats.py`) |
| H3 экспорт классификаций | `export.export_classifications_to_*()` |

## 6. Конфигурация и секреты

- Все числовые/строковые параметры — в `configs/example.yaml` (безопасно коммитить).
- Личные значения (id GEE-проекта, корневой путь на Drive) — в `configs/local.yaml`
  (в `.gitignore`) или через переменные окружения / Colab Secrets.
- `config.py` загружает YAML, валидирует обязательные поля и строит пути от
  корня проекта (перенос логики A3).

## 7. CLI

Единая точка входа `mlc` (в `pyproject.toml`, `project.scripts`):

- `mlc run --config configs/local.yaml` — полный прогон целевых лет (H1).
- `mlc diagnostics --config ...` — доступность снимков и диагностика (F1, F2).
- `mlc export --config ...` — экспорт классификаций (H3).
- `mlc report --config ...` — сводка метрик (H2).

Команды тонкие: делают только парсинг аргументов и вызывают функции модулей,
чтобы ту же логику можно было вызвать из Colab/ноутбука напрямую.

## 8. Colab-драйвер

`notebooks/colab_runner.ipynb` — несколько ячеек:

```python
!git clone https://github.com/maxlansky1/moscow-landcover-analysis.git
%cd moscow-landcover-analysis && !pip install -e .
from google.colab import drive; drive.mount('/content/drive')
import ee; ee.Authenticate(); ee.Initialize(project=USERDATA['EE_PROJECT'])
from moscow_landcover.pipeline import run
run(config_path='configs/local.yaml')
```

- Первая ячейка при повторной сессии — `!git pull` (Colab не тянет автоматически).
- Интерактив (карты, графики, отдельные проверки) — в `notebooks/exploration.ipynb`,
  который вызывает функции пакета.

## 9. Стратегия валидации

Главный риск рефакторинга — **изменение численных результатов**. Меры:

1. Перед началом сохранить эталон: `notebooks/legacy_reference.ipynb` (текущая
   версия ноутбука) и, где возможно, зафиксировать выходные метрики (H2) как
   «золотой» набор.
2. После каждого вынесенного модуля сверять результаты с эталоном на тех же
   параметрах (те же годы, территории, признаки).
3. Юнит-тесты на детерминированные функции: гармонизация имён каналов,
   маски, построение путей, загрузка конфига, парсинг важности признаков.
4. Дымовой тест: `mlc run` на сокращённом наборе (1 год × 1 территория) в Colab.

## 10. Фазы работ (чеклист)

> **Статус.** Фазы 0–7 выполнены (единым коммитом в `main`, без ветки
> `refactor/package` — по договорённости работали в `main`). Фаза 8 требует
> прогона в Colab с реальным GEE: `ee`/`geemap`/`matplotlib`/`scipy` в
> серверном окружении не установлены, поэтому локально проверены только
> импорт, синтаксис и чистая логика (юнит-тесты).

### Фаза 0. Каркас репозитория
- [x] Инициализировать структуру каталогов (см. раздел 4).
- [x] `pyproject.toml` с зависимостями и `project.scripts = { mlc = "moscow_landcover.cli:main" }`.
- [x] `requirements.txt` с пиннингом; `requires-python = ">=3.10"`.
- [x] `.gitignore`, `LICENSE` (MIT), `README.md`.
- [x] `configs/example.yaml`.
- [ ] Сохранить `notebooks/legacy_reference.ipynb` как эталон (отложено — см. раздел 9).

### Фаза 1. Инфраструктура пакета
- [x] `logging.py` (из A4) и `__init__.py`.
- [x] `config.py`: модели `AnalysisConfig`, `MapConfig`, `StatsConfig`, построение путей (A3, C1–C3).
- [x] `test_config.py`.

### Фаза 2. GEE и границы
- [x] `gee.py`: `authenticate`, `initialize`, `check` (B1–B3).
- [x] `boundaries.py`: загрузка GeoJSON (D1).
- [x] Тесты границ (через `test_imports.py`; отдельный `test_boundaries.py` не создан).

### Фаза 3. Предобработка и композит
- [x] `preprocess.py`: маски, гармонизация, калибровка (E1).
- [x] `composite.py`: `create_composite` (E3).
- [x] `test_preprocess.py`; сверка E2/E4 с эталоном (отложено, см. раздел 9).

### Фаза 4. Точки, коллекции, модель
- [x] `points.py`: обучающие точки ESA WorldCover (E5).
- [x] `landsat.py`: `build_collection_for_year` (E7).
- [x] `model.py`: обучение классификатора, `parse_feature_importance` (E7, H1).
- [ ] Сверка E6 с эталоном (отложено, см. раздел 9).

### Фаза 5. Пайплайн, статистики, экспорт
- [x] `pipeline.py`: `process_year`, `run` (H1).
- [x] `stats.py`: форматтеры и метрики (C3) + `report.summary` (H2).
- [x] `export.py`: экспорт классификаций (H3).

### Фаза 6. Диагностика, HLS, визуализация
- [x] `diagnostics.py`: доступность и диагностика (F1, F2).
- [x] `experiments/hls.py`: тест HLS.
- [x] `viz.py`: карты и графики (B3, C2, D2).

### Фаза 7. CLI и ноутбуки
- [x] `cli.py` и точка входа `mlc`.
- [x] `notebooks/colab_runner.ipynb` (драйвер).
- [ ] `notebooks/exploration.ipynb` (интерактив) — запланирован, не создан.
- [x] README с инструкцией.

### Фаза 8. Финальная проверка
- [ ] Полный прогон в Colab, сверка метрик H2 с эталоном.
- [ ] Проверка экспорта результатов в папку на Google Диске.
- [ ] `nbstripout --install`; удалить `legacy_reference.ipynb` из `main`.
  (nbstripout не установлен; активный фильтр `filter=nbstripout` в
  `.gitattributes` намеренно НЕ включён, чтобы не ломать коммиты.)
- [ ] Тег релиза `v1.0`.

## 11. Git-процесс

- Работать в ветке `refactor/package`.
- Коммиты по фазам (по одной логической единице на фазу), понятные сообщения.
- Ноутбуки коммитить только через `nbstripout` (без выводов).
- По завершении — Pull Request / merge в `main`, тег `v1.0`.

## 12. Риски и меры

| Риск | Мера |
|---|---|
| Изменение численных результатов | Сверка с эталонным ноутбуком на каждом шаге; юнит-тесты |
| Colab не тянет git автоматически | Ячейка `!git pull` первой в драйвере |
| Секреты попадут в репозиторий | Только `example.yaml`; реальные значения в Secrets/.env; `.gitignore` |
| Пакет «тяжёлый» из-за geemap/numpy | Зависимости отдельно от dev-зависимостей; ставить только нужное в Colab |
| Кол-во строк на модуль | Держать модули небольшими; блок F2 (915 строк) разбить на функции |

## 13. Критерии готовности (Definition of Done)

- [ ] Ноутбук разобран на модули; прямой логики в ноутбуках почти нет.
- [ ] `mlc run/diagnostics/export/report` работают в Colab.
- [ ] Пакет импортируется без ошибок, тесты проходят.
- [ ] Метрики совпадают с эталонными результатами.
- [ ] В репозитории нет секретов и выводов ячеек.
- [ ] README позволяет стороннему пользователю запустить проект в Colab «с нуля».

## 14. Открытые вопросы

- Итоговый набор «золотых» метрик для сверки (какие годы/территории считать эталоном).
- Куда именно на Google Диске экспортируются результаты (точный путь папки).
- Нужен ли `exploration.ipynb` в публичном репозитории или он остаётся локальным.
