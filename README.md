# Kaggle LTV

## Содержание репозитория

```text
.
├── data/                              # локальные данные, не коммитятся
├── src/
│   ├── EDA.ipynb                      # EDA и baseline-анализ
│   ├── train.py                       # CLI одного обучения
│   ├── run_experiments.py             # запуск серии конфигураций
│   └── kaggle_ltv/                    # переиспользуемые модули пайплайна
├── tests/                             # unit-тесты и проверки notebook
├── .github/workflows/ci.yml           # CI: типизация и тесты
├── .github/workflows/experiments.yml  # ручной remote-запуск
├── Makefile
├── pyproject.toml
└── poetry.lock
```

## Требования

- Python 3.12+;
- Git;
- Poetry 2.x;
- для удалённых запусков — Linux-сервер и GitHub self-hosted runner.

## Установка локально

```bash
git clone https://github.com/gshjis/Bank-LTV.git
cd Bank-LTV
poetry install --with dev
```

Проверка окружения:

```bash
make check
make typecheck
make test
```

## Данные

Данные скачать со страницы [Findata Response на Kaggle](https://www.kaggle.com/competitions/findata-response/data).

Для локальной работы файлы должны находиться в [`data/`](data/):

```text
data/
├── response_train.csv
└── response_test.csv
```

Данные не следует коммитить в Git.

Для удалённого self-hosted runner данные хранятся вне GitHub Actions workspace:

```text
~/data/response_train.csv
~/data/response_test.csv
```

Передать их на сервер можно с локального ПК:

```bash
ssh <user_name>@IP_address "mkdir -p ~/data"
scp data/response_train.csv data/response_test.csv \
  <user_name>@IP_address:~/data/
```

## EDA

Исследование находится в [`src/EDA.ipynb`](src/EDA.ipynb). Notebook содержит:

- анализ типов и пропусков;
- оценку доли положительного target по категориям;
- поиск сильных пар категориальных признаков;
- анализ числовых распределений и выбросов;
- baseline CatBoost;
- сравнение class weights и threshold.

Notebook можно открыть локально через Jupyter или VS Code. EDA не запускается в ночном experiment workflow.

## Локальное обучение

Один запуск:

```bash
make train RUN_NAME=baseline
```

Ручной запуск серии:

```bash
make experiment MAX_RUNS=3 EXPERIMENT_NAME=kaggle-ltv-local
```

Основные параметры:

```bash
make experiment \
  EXPERIMENT_NAME=kaggle-ltv-local \
  MAX_RUNS=72 \
  START_INDEX=0 \
  SEED=42
```

Для локальных runs по умолчанию используется SQLite-база `mlflow.db`.

## MLflow локально

Запустить UI:

```bash
make mlflow-ui
```

Открыть [http://127.0.0.1:5000](http://127.0.0.1:5000).

MLflow-данные по умолчанию хранятся в [`mlflow.db`](mlflow.db), артефакты — в `mlartifacts/`.

## Тесты и типизация

```bash
make test       # pytest
make typecheck  # Pyright
make check      # py_compile
```

Тесты проверяют preprocessing, split, threshold, метрики и корректность структуры [`src/EDA.ipynb`](src/EDA.ipynb). Тяжёлое обучение и MLflow в тестах не запускаются.

## GitHub Actions

### CI

[`ci.yml`](.github/workflows/ci.yml) запускается на `push` и `pull_request`. Он выполняет:

1. установку Python, Poetry и зависимостей;
2. проверку [`poetry.lock`](poetry.lock);
3. Pyright;
4. pytest.

CI не обучает модели.

### Training experiments

[`experiments.yml`](.github/workflows/experiments.yml) запускается вручную:

1. открыть **GitHub → Actions → Training experiments**;
2. нажать **Run workflow**;
3. указать `experiment_name`, `max_runs`, `start_index` и `seed`;
4. запустить workflow.

Workflow выполняется только владельцем репозитория и только на self-hosted runner с labels:

```text
self-hosted
linux
x64
training
```

Внутри workflow:

1. checkout-ится выбранный commit;
2. устанавливаются зависимости;
3. автоматически поднимается MLflow Server на `127.0.0.1:5000`, если он ещё не запущен;
4. проверяется наличие `~/data/response_train.csv`;
5. запускается [`make experiment`](Makefile:39);
6. результаты записываются в MLflow;
7. summary загружается в GitHub Actions Artifact.

При ошибке отдельного запуска используется `STOP_ON_ERROR=1`, поэтому workflow завершается с ошибкой.

## Настройка self-hosted runner

На сервере установить GitHub Actions runner через **Repository -> Settings -> Actions -> Runners -> New self-hosted runner**.

Runner должен быть зарегистрирован с labels:

```text
self-hosted
linux
x64
training
```

После регистрации проверить его статус `Idle` или `Online` в настройках репозитория.

Код вручную в workspace копировать не нужно: [`actions/checkout`](.github/workflows/experiments.yml:40) скачивает нужный commit автоматически. Данные хранятся отдельно в `/home/gshjis/data`.

## MLflow Tracking URI и Secret

В GitHub создать **Settings -> Secrets and variables -> Actions -> New repository secret**:

```text
Name: MLFLOW_TRACKING_URI
Value: http://127.0.0.1:5000
```

Такой URI подходит, если self-hosted runner и MLflow Server работают на одном сервере.

MLflow Server хранит данные вне workspace:

```text
~/mlflow/mlflow.db
~/mlflow/artifacts/
~/mlflow/server.log
```

## Просмотр MLflow с локального ПК

MLflow слушает только `127.0.0.1` на удалённом сервере. Для доступа с ПК создать SSH-туннель:

```bash
ssh -N -L 5000:127.0.0.1:5000 <user_name>@IP_address
```

Оставить терминал открытым и перейти в браузере на:

```text
http://127.0.0.1:5000
```

## Makefile

[`Makefile`](Makefile) содержит команды:

```bash
make help                 # список команд
make install              # установка зависимостей
make check                # проверка синтаксиса
make typecheck            # проверка типов
make test                 # запуск тестов
make train                # один run
make experiment           # серия экспериментов
make experiment-dry-run   # конфигурации без запуска
make mlflow-ui             # локальный MLflow UI
make mlflow-server         # MLflow Tracking Server
```

## Где искать результаты

- метрики и параметры — в MLflow experiment;
- модели и preprocessing — в MLflow artifacts;
- локальные файлы одного запуска — в `artifacts/`;
- сводка серии — `artifacts/experiments/batch_summary.jsonl`;
- логи MLflow Server — `~/mlflow/server.log`.
