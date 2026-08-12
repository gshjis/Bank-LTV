.DEFAULT_GOAL := help

POETRY ?= poetry
PYTHON := $(POETRY) run python
MLFLOW := $(POETRY) run mlflow

TRAIN_PATH ?= data/response_train.csv
ARTIFACTS_DIR ?= artifacts
EXPERIMENT_NAME ?= kaggle-ltv-nightly
TRACKING_URI ?= sqlite:///mlflow.db
SEED ?= 42
MAX_RUNS ?= 0
START_INDEX ?= 0
PORT ?= 5000

.PHONY: help install check typecheck test train experiment experiment-dry-run mlflow-ui mlflow-server

help: ## Показать доступные команды
	@awk 'BEGIN {FS = ":.*##"; printf "Использование: make <цель> [VAR=value]\n\nЦели:\n"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Установить зависимости через Poetry
	$(POETRY) install

check: ## Проверить синтаксис Python-скриптов
	$(PYTHON) -m py_compile src/train.py src/run_experiments.py

typecheck: ## Проверить типы через Pyright
	$(POETRY) run pyright

test: ## Запустить тесты pytest
	$(POETRY) run pytest -q

train: ## Запустить один обучающий run
	$(PYTHON) src/train.py \
		--train-path $(TRAIN_PATH) \
		--artifacts-dir $(ARTIFACTS_DIR) \
		--experiment-name $(EXPERIMENT_NAME) \
		--tracking-uri $(TRACKING_URI) \
		--seed $(SEED) \
		$(if $(RUN_NAME),--run-name $(RUN_NAME),)

experiment: ## Запустить серию ночных MLflow-экспериментов
	$(PYTHON) src/run_experiments.py \
		--train-path $(TRAIN_PATH) \
		--artifacts-dir $(ARTIFACTS_DIR)/experiments \
		--experiment-name $(EXPERIMENT_NAME) \
		--tracking-uri $(TRACKING_URI) \
		--seed $(SEED) \
		--max-runs $(MAX_RUNS) \
		--start-index $(START_INDEX) \
		--shuffle

experiment-dry-run: ## Показать конфигурации серии без запуска обучения
	$(PYTHON) src/run_experiments.py \
		--train-path $(TRAIN_PATH) \
		--experiment-name $(EXPERIMENT_NAME) \
		--seed $(SEED) \
		--max-runs $(MAX_RUNS) \
		--shuffle \
		--dry-run

mlflow-ui: ## Запустить локальный MLflow UI на SQLite
	$(MLFLOW) ui \
		--backend-store-uri $(TRACKING_URI) \
		--port $(PORT)

mlflow-server: ## Запустить MLflow Tracking Server для доступа по сети
	$(MLFLOW) server \
		--host 0.0.0.0 \
		--port $(PORT) \
		--backend-store-uri $(TRACKING_URI) \
		--default-artifact-root ./mlartifacts
