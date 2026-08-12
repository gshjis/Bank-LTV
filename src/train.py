"""Обучение CatBoost без EDA и утечки данных.

Скрипт использует только размеченный train-файл. Все статистики preprocessing
вычисляются исключительно на train-части, validation используется для early
stopping и выбора порога, test — только для финальной оценки.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, TypeAlias, cast

import catboost as cb
import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

type Metrics = dict[str, Any]
type ModelParams = dict[str, Any]

DEFAULT_DROP_COLUMNS: list[str] = [
    "previous-cards",
    "client_id",
    "fact-region",
    "region",
    "registration-region",
    "tp-foreign",
    "reg-and-fact-equality",
    "post-and-fact-equality",
    "reg-and-post-equality",
    "reg-fact-post-and-last-credit-equality",
    "total-of-delinquencies",
    "max-delinquency-no",
    "mean-delinquency-amount",
    "driving-license",
    "cottage",
    "garage",
    "land",
    "reg-phone",
]


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
    parser.add_argument("--auto-class-weights", choices=["Balanced", "None"], default="Balanced")
    parser.add_argument("--experiment-name", default="kaggle-ltv")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--tracking-uri",
        default=f"sqlite:///{project_dir / 'mlflow.db'}",
        help="MLflow tracking URI; по умолчанию используется локальная SQLite БД",
    )
    parser.add_argument("--registered-model-name", default=None)
    return parser.parse_args()


def split_data(
    df: pd.DataFrame, target: str, seed: int, test_size: float, val_size: float
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Делит данные на train/validation/test со стратификацией."""
    if not 0 < test_size < 1 or not 0 < val_size < 1:
        raise ValueError("test-size и val-size должны быть между 0 и 1")
    if test_size + val_size >= 1:
        raise ValueError("Сумма test-size и val-size должна быть меньше 1")

    x = df.drop(columns=[target])
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(  # type: ignore[assignment]
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    relative_val_size = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(  # type: ignore[assignment]
        x_train,
        y_train,
        test_size=relative_val_size,
        random_state=seed,
        stratify=y_train,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test  # type: ignore[return-value]


class TabularPreprocessor:
    """Минимальный preprocessing, обучаемый только на train-выборке."""

    def __init__(self, drop_columns: list[str]) -> None:
        self.drop_columns: list[str] = drop_columns.copy()
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.numeric_medians: dict[str, float] = {}
        self.categorical_modes: dict[str, str] = {}

    @staticmethod
    def _normalize_categories(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        result = df.copy()
        for column in columns:
            result[column] = (
                result[column]
                .where(result[column].notna(), "__MISSING__")
                .astype(str)
                .str.lower()
            )
        return result

    def fit(self, df: pd.DataFrame) -> TabularPreprocessor:
        clean = df.drop(columns=self.drop_columns, errors="ignore").copy()
        self.categorical_columns = clean.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self.numeric_columns = [
            column for column in clean.columns if column not in self.categorical_columns
        ]

        clean = self._normalize_categories(clean, self.categorical_columns)
        for column in self.numeric_columns:
            numeric_series = cast(
                pd.Series, pd.to_numeric(clean[column], errors="coerce")
            )
            median_value = float(numeric_series.median())
            self.numeric_medians[column] = (
                median_value if not math.isnan(median_value) else 0.0
            )
        for column in self.categorical_columns:
            mode = clean[column].mode(dropna=False)
            self.categorical_modes[column] = (
                str(mode.iloc[0]).lower() if not mode.empty else "__MISSING__"
            )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.drop(columns=self.drop_columns, errors="ignore").copy()
        expected = self.numeric_columns + self.categorical_columns
        missing = sorted(set(expected) - set(result.columns))
        if missing:
            raise ValueError(f"В данных отсутствуют признаки: {missing}")
        result = cast(pd.DataFrame, result[expected])
        result = self._normalize_categories(result, self.categorical_columns)
        for column in self.numeric_columns:
            numeric_series = cast(
                pd.Series, pd.to_numeric(result[column], errors="coerce")
            )
            result[column] = numeric_series.fillna(self.numeric_medians[column])
        for column in self.categorical_columns:
            result[column] = result[column].fillna(self.categorical_modes[column])
        return result

    def metadata(self) -> dict[str, object]:
        return {
            "drop_columns": self.drop_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "numeric_medians": self.numeric_medians,
            "categorical_modes": self.categorical_modes,
        }


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [
        f1_score(y_true, probabilities >= t, zero_division=0)  # type: ignore[arg-type]
        for t in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def calculate_metrics(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> Metrics:
    predictions = (probabilities >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "threshold": threshold,
        "classification_report": classification_report(
            y_true,
            predictions,
            output_dict=True,
            zero_division=0,  # type: ignore[arg-type]
        ),
    }
    if y_true.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        metrics["average_precision"] = float(
            average_precision_score(y_true, probabilities)
        )
    return metrics


def flatten_metrics(metrics: Metrics) -> dict[str, float]:
    """Преобразует вложенный classification report в плоские MLflow metrics."""
    flat: dict[str, float] = {}

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for name, nested_value in value.items():
                visit(nested_value, f"{prefix}_{name}" if prefix else str(name))
        elif isinstance(value, (int, float)):
            flat[prefix] = float(value)

    visit(metrics, "")
    return flat


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

    model_params: ModelParams = {
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
