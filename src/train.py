"""Обучение CatBoost без EDA и утечки данных.

Скрипт использует только размеченный train-файл. Все статистики preprocessing
вычисляются исключительно на train-части, validation используется для early
stopping и выбора порога, test — только для финальной оценки.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import catboost as cb
import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
from kaggle_ltv.config import DEFAULT_DROP_COLUMNS
from kaggle_ltv.metrics import calculate_metrics, choose_threshold, flatten_metrics
from kaggle_ltv.preprocessing import TabularPreprocessor
from kaggle_ltv.splitting import split_data


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train CatBoost classifier")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=project_dir / "data" / "response_train.csv",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=project_dir / "artifacts",
    )
    parser.add_argument("--target", default="target")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--l2-leaf-reg", type=float, default=10.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument(
        "--auto-class-weights", choices=["Balanced", "None"], default="Balanced"
    )
    parser.add_argument("--experiment-name", default="kaggle-ltv")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--tracking-uri",
        default=f"sqlite:///{project_dir / 'mlflow.db'}",
        help="MLflow tracking URI; по умолчанию используется локальная SQLite БД",
    )
    parser.add_argument("--registered-model-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    df = pd.read_csv(args.train_path)
    if args.target not in df.columns:
        raise ValueError(f"В train-файле отсутствует target: {args.target}")
    target_values = cast(pd.Series, df[args.target])
    if bool(target_values.isna().any()) or not set(target_values.unique()).issubset({0, 1}):
        raise ValueError("Поддерживается только бинарный target со значениями 0 и 1")

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(
        df, args.target, args.seed, args.test_size, args.val_size
    )
    preprocessor = TabularPreprocessor(DEFAULT_DROP_COLUMNS).fit(x_train)
    x_train = preprocessor.transform(x_train)
    x_val = preprocessor.transform(x_val)
    x_test = preprocessor.transform(x_test)

    model_params: dict[str, Any] = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "auto_class_weights": None if args.auto_class_weights == "None" else "Balanced",
        "eval_metric": "AUC",
        "early_stopping_rounds": args.early_stopping_rounds,
        "random_seed": args.seed,
        "verbose": 100,
        "allow_writing_files": False,
    }
    logged_model_params = {
        **model_params,
        "auto_class_weights": args.auto_class_weights,
    }

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                **logged_model_params,
                "train_path": str(args.train_path),
                "target": args.target,
                "test_size": args.test_size,
                "val_size": args.val_size,
                "train_rows": len(x_train),
                "validation_rows": len(x_val),
                "test_rows": len(x_test),
                "numeric_features": len(preprocessor.numeric_columns),
                "categorical_features": len(preprocessor.categorical_columns),
                "dropped_features": len(DEFAULT_DROP_COLUMNS),
            }
        )
        mlflow.set_tags(
            {
                "framework": "catboost",
                "pipeline": "train-only-preprocessing",
                "split_strategy": "stratified-train-validation-test",
            }
        )

        model = cb.CatBoostClassifier(**model_params)
        model.fit(
            x_train,
            y_train,
            eval_set=(x_val, y_val),
            cat_features=preprocessor.categorical_columns,
        )

        val_probabilities = model.predict_proba(x_val)[:, 1]
        threshold = choose_threshold(y_val, val_probabilities)
        test_probabilities = model.predict_proba(x_test)[:, 1]
        metrics = {
            "validation": calculate_metrics(y_val, val_probabilities, threshold),
            "test": calculate_metrics(y_test, test_probabilities, threshold),
            "best_iteration": int(model.get_best_iteration()),
            "seed": args.seed,
        }
        mlflow.log_metrics(flatten_metrics(metrics))

        model_path = args.artifacts_dir / "model.cbm"
        preprocessor_path = args.artifacts_dir / "preprocessor.json"
        metrics_path = args.artifacts_dir / "metrics.json"
        model.save_model(str(model_path))
        preprocessor_path.write_text(
            json.dumps(preprocessor.metadata(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        mlflow.log_artifacts(str(args.artifacts_dir), artifact_path="artifacts")
        model_log_kwargs: dict[str, Any] = {"artifact_path": "model"}
        if args.registered_model_name:
            model_log_kwargs["registered_model_name"] = args.registered_model_name
        mlflow.catboost.log_model(model, **model_log_kwargs)  # type: ignore[attr-defined]

        print(f"MLflow run: {run.info.run_id}")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
